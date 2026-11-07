"""Edit mode: the harness drives, the model only writes the fix.

Small local models (1-3B) cannot run an open-ended tool loop reliably: they pick
the wrong file, invent arguments, or repeat themselves. In edit mode the harness
does everything except the actual change:

  1. run the tests and keep the failure,
  2. pick the files (from the traceback, then the request),
  3. show them to the model and ask for SEARCH/REPLACE edit blocks,
  4. apply the blocks (exact, then whitespace-tolerant), show diffs, ask approval,
  5. re-run the tests and feed the new failure back, up to ``rounds`` times.

One clear decision per round is a task a small model can do.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

EDIT_SYSTEM = """You are Ablate Code. You fix and change code by writing edit blocks, in exactly this format:

app/util.py
<<<<<<< SEARCH
    return a - b
=======
    return a + b
>>>>>>> REPLACE

Rules:
- The first line is the file path. SEARCH lines must be copied exactly from the file shown, with the same indentation.
- Keep SEARCH short: the lines you change plus one line around them. Use several blocks for several changes.
- To create a new file, leave SEARCH empty.
- Change the code, not the tests, unless the user asks for test changes.
- After the blocks, write one short sentence saying what you changed."""

BLOCK = re.compile(
    r"(?:(?P<path>[^\n]*)\n)?(?:```[\w+-]*\n)?<{5,}\s*SEARCH\s*\n(?P<search>.*?)\n?={5,}\s*\n(?P<replace>.*?)\n?>{5,}\s*REPLACE",
    re.S)
TRACE = [re.compile(r'File "([^"]+)", line (\d+)'),            # Python tracebacks
         re.compile(r"^([\w./-]+\.\w+):(\d+)(?::\d+)?[: ]", re.M),  # file.py:12: / file.ts:3:4
         re.compile(r"at .*?\(?([\w./-]+\.\w+):(\d+):\d+\)?")]   # JS stacks


def parse_blocks(text: str, default_path: Optional[str]) -> List[Tuple[str, str, str]]:
    out = []
    for m in BLOCK.finditer(text):
        path = (m.group("path") or "").strip().strip("`*:#").strip()
        path = re.sub(r"^(file|path)\s*[:=]\s*", "", path, flags=re.I).strip()
        if not path or " " in path or "<" in path:
            path = default_path or ""
        out.append((path, m.group("search"), m.group("replace")))
    return out


def outside_blocks(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", BLOCK.sub("", text)).replace("```", "").strip()


class Editor:
    def __init__(self, agent) -> None:
        self.a = agent
        self.ws, self.ui, self.be, self.cfg = agent.ws, agent.ui, agent.be, agent.cfg

    # -- context -------------------------------------------------------------------- #
    def pick_files(self, request: str, tests: str, limit: int = 2) -> List[str]:
        """Source files to show: traceback frames first (non-test, inside the project), then the request."""
        files = set(self.ws.files())
        found: List[str] = []

        def add(f: str) -> None:
            try:
                rel = self.ws.rel(self.ws.path(f))
            except ValueError:
                return
            if rel in files and rel not in found:
                found.append(rel)

        frames = []
        for rx in TRACE:
            frames += [m.group(1) for m in rx.finditer(tests or "")]
        for f in frames:  # innermost frames come last in Python tracebacks; prefer non-test code
            if "test" not in f.rsplit("/", 1)[-1].lower():
                add(f)
        for line in self.ws.relevant(request).splitlines():
            f = line.split(" (file name", 1)[0].split(":", 1)[0]
            if "test" not in f.rsplit("/", 1)[-1].lower():
                add(f)
        for mention in re.findall(r"[\w./-]+\.\w{1,5}", request):
            add(mention)
        if not found:  # nothing obvious: the failing test file at least shows what is expected
            for f in frames:
                add(f)
        return found[:limit]

    def show(self, path: str, budget_chars: int) -> str:
        text = (self.ws.root / path).read_text(encoding="utf-8", errors="replace")
        if len(text) > budget_chars:
            text = text[:budget_chars] + "\n… (file truncated; ask for the rest if you need it)"
        return f"{path}\n```\n{text}\n```"

    # -- one request ------------------------------------------------------------------- #
    def run(self, request: str) -> str:
        ui, ws = self.ui, self.ws
        test_cmd = self.a.test_cmd
        rounds = int(self.cfg.get("edit_rounds", 3))
        tests = ""
        if test_cmd:
            ui.tool("Bash", f"{test_cmd}  (checking the current state)")
            ui.spinner("running tests")
            tests = ws.bash(test_cmd)
            ui.result(tests, ok=tests.startswith("exit code 0"), max_lines=4)
        files = self.pick_files(request, tests)
        if not files:
            return self.a.agent_ask(request, note="(Edit mode found no file to show, so this ran as a normal agent turn.)")
        feedback, applied_any, summary = "", False, ""
        for rnd in range(1, rounds + 1):
            per_file = max(2000, (int(self.a.budget) - int(self.cfg["max_new_tokens"]) - 900) * 3 // max(1, len(files)))
            for f in files:
                ui.tool("Read", f"{f}  (shown to the model)")
            parts = [f"Task: {request}"]
            if tests and not tests.startswith("exit code 0"):
                parts.append("Test output:\n" + "\n".join(tests.splitlines()[-30:]))
            if feedback:
                parts.append(feedback)
            parts.append("Files:\n\n" + "\n\n".join(self.show(f, per_file) for f in files))
            parts.append("Reply with edit blocks.")
            msgs = [{"role": "system", "content": EDIT_SYSTEM + self.extra()}, {"role": "user", "content": "\n\n".join(parts)}]
            ui.spinner("writing the fix" if rnd == 1 else f"round {rnd}")
            shown: List[str] = []
            text, _, _ = self.be.generate(msgs, None, {"temperature": self.cfg["temperature"],
                                                       "max_new_tokens": self.cfg["max_new_tokens"],
                                                       "top_p": 0.95, "repetition_penalty": 1.0}, shown.append)
            blocks = parse_blocks(text, files[0])
            summary = outside_blocks(text)
            if not blocks:
                if rnd == rounds:
                    break
                feedback = ("Your reply had no edit blocks, so nothing changed. Reply with blocks in the exact "
                            "SEARCH/REPLACE format from the instructions.")
                ui.info("(no edit blocks in the reply, asking again)")
                continue
            errors = []
            for path, search, replace in blocks:
                ok, msg = self.apply(path, search, replace)
                applied_any |= ok
                if not ok:
                    errors.append(msg)
            if not test_cmd:
                break
            ui.tool("Bash", test_cmd)
            ui.spinner("running tests")
            tests = ws.bash(test_cmd)
            passed = tests.startswith("exit code 0")
            ui.result(tests, ok=passed, max_lines=4)
            if passed and applied_any:
                return self.finish(request, summary or "Fixed; the tests pass.", True)
            feedback = ""
            if errors:
                feedback += "Some edit blocks did not apply:\n" + "\n".join(errors[:3]) + "\n\n"
            feedback += "The tests still fail after your change. Fix the remaining problem. The files below are the current versions."
        return self.finish(request, summary or "I could not complete the change.", False)

    def extra(self) -> str:
        ins = self.ws.instructions(1200)
        return ("\n\n" + ins) if ins else ""

    def apply(self, path: str, search: str, replace: str) -> Tuple[bool, str]:
        ui, ws = self.ui, self.ws
        try:
            target = ws.path(path)
        except ValueError as e:
            return False, str(e)
        if not search.strip():
            if target.exists() and target.read_text(encoding="utf-8", errors="replace").strip():
                return False, f"{path}: SEARCH was empty but the file already has content."
            ui.tool("Write", path)
            msg, change = ws.write_file(path, replace + ("\n" if not replace.endswith("\n") else ""))
        else:
            ui.tool("Update", path)
            msg, change = ws.edit_file(path, search, replace)
        if not change:
            ui.result(msg.splitlines()[0], ok=False)
            return False, msg
        ui.diff(path, *change)
        ok, why = self.a.permit_edit(path)
        if not ok:
            ws.revert()
            ui.result("rejected", ok=False)
            return False, why
        ui.result(msg)
        return True, msg

    def finish(self, request: str, summary: str, ok: bool) -> str:
        self.ui.out("")
        self.ui.out(self.ui.c("ok" if ok else "warn", "✓ " if ok else "• ") + summary)
        self.a.turns.append(f"- user: {request[:140]}  →  {'done' if ok else 'not finished'}: {summary[:160]}")
        self.a.history += [{"role": "user", "content": request}, {"role": "assistant", "content": summary}]
        self.a.save()
        return summary
