"""The ``ablate code`` agent loop.

model picks a tool -> harness checks permission -> runs it -> result goes back
to the model -> repeat until the model answers without a tool call.

Everything here is shaped by one constraint: the models are small (1-3B) with a
short usable context. So the harness does the work a big model would do itself:
it maps the repo, pre-searches for what the request mentions, keeps tool output
short, hides stale results, compacts history deterministically (never losing
the user's goal), and keeps the model's plan in an on-disk scratchpad.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import config as C
from . import skills as SK
from .mcp import MCPHub
from .tools import READ_ONLY, SCHEMAS, Workspace, command_allowed, is_dangerous
from .ui import UI

KEEP_RECENT = 6          # messages whose tool results stay in full
STUB_OVER = 320          # older tool results longer than this become one-line stubs

SYSTEM = """You are Ablate Code, a coding agent working inside the user's project. You act by calling tools; each result comes back to you.

How to work:
1. Find the relevant code with grep, list_files and read_file. Read before you edit.
2. Make the smallest correct change with edit_file. Copy 'old' exactly as the file shows it, without line numbers.
3. Check your work: run the tests or the program with bash.
4. When the task is done, reply in plain text with a short summary of what you changed. Do not call a tool in that reply.

Rules: call one tool per reply. Never guess file contents. If a tool returns an error, read it and fix your call. For tasks with several steps, save your plan with note."""


def _calls(text: str, names: set) -> Tuple[List[dict], str]:
    """Pull tool calls out of model text. Accepts <tool_call>{...}</tool_call>, a bare JSON object,
    or a ```json fenced block, since small models use all three."""
    found: List[dict] = []
    spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", text, re.S):
        found.append(m.group(1))
        spans.append(m.span())
    if not found:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        if m and '"name"' in m.group(1):
            found.append(m.group(1))
            spans.append(m.span())
        elif text.strip().startswith("{"):
            found.append(text.strip())
            spans.append((0, len(text)))
    calls = []
    for blob in found:
        obj = None
        for cand in (blob, blob[: blob.rfind("}") + 1]):
            try:
                obj = json.loads(cand)
                break
            except ValueError:
                continue
        if not isinstance(obj, dict):
            continue
        fn = obj.get("function")
        name = obj.get("name") or (fn.get("name") if isinstance(fn, dict) else fn)
        args = obj.get("arguments", obj.get("parameters"))
        if args is None and isinstance(fn, dict):
            args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        if name in names:
            calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
    prose = text
    for a, b in reversed(spans):
        prose = prose[:a] + prose[b:]
    return calls, prose.strip()


class StreamFilter:
    """Show the model's prose live, but never its raw tool-call JSON."""

    def __init__(self, out: Callable[[str], None]) -> None:
        self.out, self.buf, self.shown, self.mode = out, "", 0, None

    def feed(self, piece: str) -> None:
        self.buf += piece
        if self.mode is None:
            head = self.buf.lstrip()
            if len(head) < 12 and ("<tool_call>".startswith(head) or head.startswith(("{", "`"))):
                return
            self.mode = "hide" if head.startswith(("<tool_call>", "{", "```json")) else "text"
        if self.mode == "text":
            cut = self.buf.find("<tool_call>")
            end = cut if cut >= 0 else max(self.shown, len(self.buf) - 11)
            if end > self.shown:
                self.out(self.buf[self.shown:end])
                self.shown = end
            if cut >= 0:
                self.mode = "hide"

    def flush(self) -> None:
        if self.mode in (None, "text"):
            rest = self.buf[self.shown:]
            if "<tool_call>" not in rest and not rest.lstrip().startswith(("{", "```json")):
                self.out(rest)
            self.shown = len(self.buf)


class Agent:
    def __init__(self, root: Path, cfg: dict, ui: UI, backend) -> None:
        self.root, self.cfg, self.ui, self.be = root, cfg, ui, backend
        self.ws = Workspace(root)
        self.skills = SK.find(root)
        self.mcp = MCPHub(cfg.get("mcp", {}).get("servers", {}))
        self.history: List[dict] = []
        self.turns: List[str] = []           # one-line summaries of finished requests
        self.session_ok: set = set()         # commands approved with "always"
        self.auto_edits = cfg["mode"] in ("auto", "yolo")
        self.test_cmd = cfg.get("test_command") or self.ws.detect_tests()
        self.ws.test_cmd = self.test_cmd
        self.repo = self.ws.repo_map()
        self.budget = min(int(cfg["context"]), backend.max_context())
        self.session_file = C.USER_DIR / "sessions" / hashlib.sha1(str(root).encode()).hexdigest()[:12] / \
            (time.strftime("%Y%m%d-%H%M%S") + ".json")

    # -- prompt ------------------------------------------------------------------ #
    def tools(self) -> List[dict]:
        names = ["list_files", "grep", "read_file", "edit_file", "write_file", "bash", "note"]
        if self.cfg["mode"] == "plan":
            names = ["list_files", "grep", "read_file", "bash", "note"]
        if self.skills:
            names.append("skill")
        if self.mcp:
            names.append("mcp")
        return [SCHEMAS[n] for n in names]

    def system(self) -> str:
        now = _dt.datetime.now().astimezone()
        parts = [SYSTEM, f"You are in the project folder {self.root.name}/. Use paths relative to it, like "
                         f"src/main.py.\nDate: {now:%A %d %B %Y}"]
        if self.cfg["mode"] == "plan":
            parts.append("PLAN MODE: you may only read and run read-only commands. Investigate, then reply with a plan.")
        if self.test_cmd:
            parts.append(f"Test command: {self.test_cmd}")
        parts.append("Project files:\n" + self.repo)
        ins = self.ws.instructions()
        if ins:
            parts.append(ins)
        if self.skills:
            parts.append("Skills (load one with the skill tool when it fits the task):\n" + SK.listing(self.skills))
        if self.mcp:
            parts.append("MCP servers (use the mcp tool; call it with only 'server' to see its tools): " + ", ".join(self.mcp.names()))
        if self.turns:
            parts.append("Earlier in this session:\n" + "\n".join(self.turns[-6:]))
        if self.ws.log:
            parts.append("Your recent actions:\n" + "\n".join(f"- {a}" for a in self.ws.log[-10:]))
        pad = self.ws.scratchpad()
        if pad:
            parts.append("Your scratchpad (notes you saved):\n" + pad)
        return "\n\n".join(parts)

    # -- context management ------------------------------------------------------------ #
    def build(self, start: int) -> Tuple[List[dict], int]:
        """Messages to send, squeezed into the budget. ``start`` is where the current request begins."""
        reserve = int(self.cfg["max_new_tokens"])
        tools = self.tools()

        def stub(msgs: List[dict], keep: int, limit: int) -> List[dict]:
            out = []
            for i, m in enumerate(msgs):
                if m["role"] == "tool" and i < len(msgs) - keep and len(m["content"]) > limit:
                    first = m["content"].strip().splitlines()[0][:120]
                    m = {**m, "content": f"{first} … [older result hidden to save space; repeat the call if you need it]"}
                out.append(m)
            return out

        current = self.history[start:]
        for keep, limit in ((KEEP_RECENT, STUB_OVER), (4, 120), (2, 80)):
            msgs = [{"role": "system", "content": self.system()}] + stub(current, keep, limit)
            used = self.be.count(msgs, tools)
            if used + reserve <= self.budget:
                return msgs, used
        # Still too big: keep the request and the latest steps, drop the middle. The actions log in the
        # system prompt and the scratchpad carry what was learned.
        head, tail = current[:1], current[1:]
        while len(tail) > 2:
            tail = tail[2:]
            while tail and tail[0]["role"] == "tool":
                tail = tail[1:]
            note = {"role": "user", "content": "[Earlier steps for this request were removed to fit the context. "
                                               "See 'Your recent actions' and your scratchpad. Continue the task.]"}
            msgs = [{"role": "system", "content": self.system()}] + head + [note] + stub(tail, 2, 80)
            used = self.be.count(msgs, tools)
            if used + reserve <= self.budget:
                return msgs, used
        return msgs, used

    # -- permissions --------------------------------------------------------------------- #
    def _same_as_tests(self, cmd: str) -> bool:
        """True when cmd is the project test command, even with the quoting lost."""
        if not self.test_cmd:
            return False
        import shlex
        try:
            want = shlex.split(self.test_cmd)
        except ValueError:
            return cmd == self.test_cmd
        return cmd == self.test_cmd or cmd.split() == " ".join(want).split()

    def permit_command(self, cmd: str) -> Tuple[bool, str]:
        mode = self.cfg["mode"]
        verdict = command_allowed(cmd, self.cfg["allow"], self.cfg["deny"])
        if verdict is False:
            return False, "This command is on the deny list."
        if cmd in self.session_ok or (self._same_as_tests(cmd) and not is_dangerous(cmd)):
            return True, ""
        if verdict is True and not is_dangerous(cmd):
            return True, ""
        if mode == "plan":
            return False, "Plan mode only allows read-only commands from the allow list."
        if mode == "yolo" and not is_dangerous(cmd):
            return True, ""
        ans = self.ui.ask(("Run a potentially destructive command: " if is_dangerous(cmd) else "Run ") +
                          self.ui.c("bold", cmd) + " ?")
        if ans == "a":
            self.session_ok.add(cmd)
            return True, ""
        return (True, "") if ans == "y" else (False, "The user declined to run this command.")

    def permit_edit(self, path: str) -> Tuple[bool, str]:
        if self.cfg["mode"] == "plan":
            return False, "Plan mode: no edits."
        if self.auto_edits:
            return True, ""
        ans = self.ui.ask(f"Apply this change to {self.ui.c('bold', path)}?")
        if ans == "a":
            self.auto_edits = True
            return True, ""
        return (True, "") if ans == "y" else (False, "The user rejected this edit. Ask what they want instead, or try another approach.")

    # -- tool dispatch ------------------------------------------------------------------- #
    def run_tool(self, name: str, a: dict) -> str:
        ws, ui = self.ws, self.ui
        params = SCHEMAS[name]["function"]["parameters"]
        missing = [k for k in params.get("required", []) if not str(a.get(k, "")).strip()]
        unknown = [k for k in a if k not in params.get("properties", {})]
        if missing:
            ui.tool(name, "invalid call")
            msg = (f"Error: {name} needs {', '.join(repr(m) for m in missing)}. Its parameters are: "
                   f"{', '.join(params['properties'])}." + (f" Unknown: {', '.join(map(repr, unknown))}." if unknown else ""))
            ui.result(msg, ok=False)
            return msg
        a = {k: v for k, v in a.items() if k in params.get("properties", {})}
        try:
            if name == "list_files":
                ui.tool("List", a.get("path", ".") + (f", {a['pattern']}" if a.get("pattern") else ""))
                r = ws.list_files(a.get("path", "."), a.get("pattern", "*"))
                ui.result(r.splitlines()[0])
                return r
            if name == "grep":
                ui.tool("Search", f"{a.get('pattern', '')!r}" + (f" in {a['path']}" if a.get("path") else ""))
                r = ws.grep(str(a.get("pattern", "")), a.get("path", "."), a.get("glob", ""))
                n = 0 if r.startswith("No matches") else len(r.splitlines())
                ui.result(f"{n} matches" if n else r)
                return r
            if name == "read_file":
                rng = f":{a.get('start', 1)}-{a['end']}" if a.get("end") else (f":{a['start']}+" if a.get("start") else "")
                ui.tool("Read", f"{a.get('path', '')}{rng}")
                r = ws.read_file(str(a.get("path", "")), a.get("start", 1), a.get("end"))
                ui.result(r if r.startswith("Error") else f"{len(r.splitlines())} lines", ok=not r.startswith("Error"))
                return r
            if name in ("edit_file", "write_file"):
                path = str(a.get("path", ""))
                ui.tool("Update" if name == "edit_file" else "Write", path)
                if name == "edit_file":
                    msg, change = ws.edit_file(path, str(a.get("old", "")), str(a.get("new", "")))
                else:
                    msg, change = ws.write_file(path, str(a.get("content", "")))
                if not change:
                    ui.result(msg.splitlines()[0], ok=False)
                    return msg
                ui.diff(path, *change)
                ok, why = self.permit_edit(path)
                if not ok:
                    ws.revert()
                    ws.log.append(f"edit to {path} rejected by user")
                    ui.result("rejected", ok=False)
                    return why
                ui.result(msg)
                return msg + (f" Next, check it: {self.test_cmd}" if self.test_cmd and name == "edit_file" else "")
            if name == "bash":
                cmd = str(a.get("command", "")).strip()
                ui.tool("Bash", cmd[:100])
                ok, why = self.permit_command(cmd)
                if not ok:
                    ui.result(why, ok=False)
                    return why
                ui.spinner("running")
                r = ws.bash(self.test_cmd if self._same_as_tests(cmd) else cmd)
                ui.result(r, ok=r.startswith("exit code 0"), max_lines=8)
                return r
            if name == "note":
                ui.tool("Note", str(a.get("text", ""))[:80])
                return ws.note(str(a.get("text", "")))
            if name == "skill":
                ui.tool("Skill", str(a.get("name", "")))
                return SK.load(self.skills, str(a.get("name", "")))
            if name == "mcp":
                srv, tool = str(a.get("server", "")), a.get("tool")
                ui.tool("MCP", f"{srv}{'.' + tool if tool else ' · list tools'}")
                if tool and self.cfg["mode"] != "yolo":
                    ok = self.ui.ask(f"Call MCP tool {self.ui.c('bold', srv + '.' + tool)}?") in ("y", "a")
                    if not ok:
                        return "The user declined this MCP call."
                r = self.mcp.run(srv, tool, a.get("arguments") or {})
                ui.result(r, ok=not r.startswith("MCP error"))
                return r
            return f"Unknown tool {name}."
        except Exception as e:  # a tool failure goes back to the model, not up the stack
            ui.result(f"{type(e).__name__}: {e}", ok=False)
            return f"Error: {type(e).__name__}: {e}"

    # -- one user request ------------------------------------------------------------------ #
    def strategy(self) -> str:
        s = self.cfg.get("strategy", "auto")
        return s if s in ("agent", "edit") else ("edit" if getattr(self.be, "small", False) else "agent")

    def ask(self, request: str) -> str:
        if self.strategy() == "edit" and self.cfg["mode"] != "plan":
            from .editor import Editor
            return Editor(self).run(request)
        return self.agent_ask(request)

    def agent_ask(self, request: str, note: Optional[str] = None) -> str:
        if note:
            self.ui.info(note)
        hints = self.ws.relevant(request)
        content = request + (f"\n\n(Possibly relevant, found by searching the project:\n{hints})" if hints else "")
        pre = self.ws.preload(hints) if hints else ""
        if pre:
            content += "\n\n" + pre
        if self.test_cmd and re.search(r"\b(fix|bug|crash|fail|broken|error|test)", request, re.I):
            self.ui.tool("Bash", self.test_cmd + "  (checking the current state)")
            self.ui.spinner("running tests")
            out = self.ws.bash(self.test_cmd)
            self.ui.result(out, ok=out.startswith("exit code 0"), max_lines=4)
            tail = "\n".join(out.splitlines()[-25:])
            content += f"\n\nTests right now ({self.test_cmd}):\n{tail}"
        start = len(self.history)
        self.history.append({"role": "user", "content": content})
        names = {s["function"]["name"] for s in self.tools()}
        seen: Dict[str, int] = {}
        final, bad = "", 0
        opts = {"temperature": self.cfg["temperature"], "max_new_tokens": self.cfg["max_new_tokens"],
                "top_p": 0.95, "repetition_penalty": 1.0}
        for step in range(int(self.cfg["max_steps"])):
            msgs, used = self.build(start)
            self.ui.spinner("thinking")
            filt = StreamFilter(self.ui.stream)
            self.ui.stream_start()
            try:
                text, native, stats = self.be.generate(msgs, self.tools(), opts, filt.feed)
            except KeyboardInterrupt:
                self.be.stop()
                self.ui.stream_end()
                self.ui.info("stopped")
                self.history.append({"role": "assistant", "content": "(stopped by the user)"})
                return ""
            filt.flush()
            self.ui.stream_end()
            calls = native if native is not None else _calls(text, names)[0]
            prose = text if native is not None else _calls(text, names)[1]
            if not calls and ("<tool_call>" in text or text.strip().startswith("{")) and bad < 3:
                bad += 1   # it meant to call a tool but the JSON was broken: say so and let it retry
                self.history.append({"role": "assistant", "content": text[:600]})
                self.history.append({"role": "user", "content": (
                    "That tool call was not valid JSON, so nothing ran. Send exactly one call like "
                    '<tool_call>{"name": "read_file", "arguments": {"path": "app/main.py"}}</tool_call>')})
                self.ui.info("(malformed tool call, asking the model to retry)")
                continue
            bad = 0
            if not calls:
                final = prose.strip()
                self.history.append({"role": "assistant", "content": final})
                break
            call = calls[0]   # one at a time keeps small models on track
            key = json.dumps([call["name"], call["arguments"]], sort_keys=True)
            seen[key] = seen.get(key, 0) + 1
            self.history.append({"role": "assistant", "content": prose, "tool_calls": [
                {"type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}]})
            if seen[key] > 2:
                result = ("You already made this exact call and got the result above. Use it, or try a "
                          "different step. If the task is finished, reply with a summary and no tool call.")
            else:
                result = self.run_tool(call["name"], call["arguments"])
            self.history.append({"role": "tool", "name": call["name"], "content": result})
        else:
            final = f"(stopped after {self.cfg['max_steps']} steps; say 'continue' to keep going)"
            self.ui.info(final)
        self.turns.append(f"- user: {request[:140]}  →  you: {final[:160] or '(no summary)'}")
        self.save()
        return final

    # -- persistence -------------------------------------------------------------------------- #
    def save(self) -> None:
        try:
            self.session_file.parent.mkdir(parents=True, exist_ok=True)
            self.session_file.write_text(json.dumps({"root": str(self.root), "model": self.be.name,
                                                     "history": self.history, "turns": self.turns,
                                                     "log": self.ws.log}, indent=1), encoding="utf-8")
        except OSError:
            pass

    def resume(self) -> bool:
        folder = self.session_file.parent
        files = sorted(folder.glob("*.json")) if folder.is_dir() else []
        if not files:
            return False
        data = json.loads(files[-1].read_text(encoding="utf-8"))
        self.history, self.turns, self.ws.log = data.get("history", []), data.get("turns", []), data.get("log", [])
        self.session_file = files[-1]
        return True


# ------------------------------------------------------------------------------------------- #
# Interactive shell
# ------------------------------------------------------------------------------------------- #
HELP = """Commands
  /help                 this list
  /clear                start a fresh conversation (the scratchpad stays)
  /context              how the token budget is being used
  /mode [ask|auto|plan|yolo]   approvals: ask for edits and commands, auto-approve edits, read-only, or approve all
  /models               pick a model: on this computer, or self-hosted (RunPod / any server)
  /strategy [auto|agent|edit]  agent: the model drives tools · edit: the harness drives, the model writes the fix
  /theme [ablate|dark|light|mono]
  /skills               list skills       /mcp [server]   list MCP servers or a server's tools
  /scratchpad [clear]   show or clear the agent's notes
  /undo                 revert the last file change
  /init                 create ABLATE.md with project notes for the agent
  /config               show settings and where they live
  /exit                 quit (Ctrl+D also works)"""


def slash(agent: Agent, line: str) -> bool:
    """Handle a /command. Returns False to exit."""
    ui = agent.ui
    cmd, _, arg = line[1:].partition(" ")
    arg = arg.strip()
    if cmd in ("exit", "quit", "q"):
        return False
    if cmd == "help":
        ui.out(HELP)
    elif cmd == "clear":
        agent.history, agent.turns = [], []
        ui.info("Conversation cleared.")
    elif cmd == "context":
        msgs = [{"role": "system", "content": agent.system()}] + agent.history
        sys_t = agent.be.count([msgs[0]], agent.tools())
        all_t = agent.be.count(msgs, agent.tools())
        ui.info(f"system prompt + tools: {sys_t:,} tokens · full history: {all_t - sys_t:,} tokens · "
                f"budget per request: {agent.budget:,} (reply reserve {agent.cfg['max_new_tokens']:,})")
        ui.info("Older tool results are hidden and history is compacted automatically when a request exceeds the budget.")
    elif cmd == "mode":
        if arg in ("ask", "auto", "plan", "yolo"):
            agent.cfg["mode"] = arg
            agent.auto_edits = arg in ("auto", "yolo")
            ui.info(f"Mode: {arg}")
        else:
            ui.info(f"Mode: {agent.cfg['mode']}  (ask · auto · plan · yolo)")
    elif cmd in ("model", "models"):
        pick_model(agent)
    elif cmd == "strategy":
        if arg in ("auto", "agent", "edit"):
            agent.cfg["strategy"] = arg
        ui.info(f"Strategy: {agent.cfg.get('strategy', 'auto')} → using {agent.strategy()} mode "
                "(auto = edit for small local models, agent for self-hosted ones)")
    elif cmd == "theme":
        if ui.set_theme(arg):
            C.save_user({"theme": arg})
            ui.info(f"Theme: {arg}")
        else:
            ui.info("Themes: ablate, dark, light, mono")
    elif cmd == "skills":
        ui.out(SK.listing(agent.skills) or "  No skills. Add .ablate/skills/<name>/SKILL.md with name and description front matter.")
    elif cmd == "mcp":
        if not agent.mcp:
            ui.info("No MCP servers. Add them to .ablate/config.json under mcp.servers.")
        elif arg:
            ui.out(agent.mcp.run(arg))
        else:
            ui.info("MCP servers: " + ", ".join(agent.mcp.names()))
    elif cmd == "scratchpad":
        if arg == "clear":
            agent.ws.scratch.unlink(missing_ok=True)
            ui.info("Scratchpad cleared.")
        else:
            ui.out(agent.ws.scratchpad(4000) or "  (empty)")
    elif cmd == "undo":
        ui.info(agent.ws.revert())
    elif cmd == "init":
        p = agent.root / "ABLATE.md"
        if p.exists():
            ui.info("ABLATE.md already exists.")
        else:
            p.write_text("# Notes for Ablate Code\n\n"
                         f"- Test command: `{agent.test_cmd or 'add your test command'}`\n"
                         "- Code style: \n- Where things live: \n- Do not touch: \n", encoding="utf-8")
            ui.info("Created ABLATE.md. Keep it short: it is sent with every request.")
    elif cmd == "config":
        ui.out(json.dumps({k: v for k, v in agent.cfg.items() if k != "api_key"}, indent=2))
        ui.info(f"user: {C.USER_DIR / 'config.json'} · project: {agent.root / '.ablate' / 'config.json'}")
    else:
        ui.info(f"Unknown command /{cmd}. Try /help.")
    return True


def pick_model(agent: Agent) -> None:
    """List local models and saved RunPod/server connections; switch to the chosen one."""
    from ..chat.server import MODELS, Engine
    from .backend import LocalBackend, remote_backend, saved_remotes
    ui = agent.ui
    probe = Engine()
    rows = []
    ui.out(ui.c("bold", "  On this computer"))
    for m in MODELS:
        have = bool(probe.local_path(m["id"]))
        rows.append(("local", m["id"]))
        mark = ui.c("ok", "●") if agent.be.name == m["id"] else " "
        ui.out(f"  {mark} {len(rows):>2}. {m['name']:<18} {m['params']:>5}  " +
               ui.c("dim", ("downloaded" if have else f"download {m['size_gb']} GB") + " · " + m["license"]))
    remotes = saved_remotes()
    ui.out(ui.c("bold", "  Self-hosted · RunPod & servers"))
    if not remotes:
        ui.out(ui.c("dim", "      none yet: connect one in ablate chat → Models → Self-hosted, or add it to config.json"))
    for r in remotes:
        rows.append(("remote", r))
        host = r["base_url"].split("//", 1)[-1].split("/", 1)[0]
        mark = ui.c("ok", "●") if agent.be.name.startswith(r["name"] + " ·") else " "
        ui.out(f"  {mark} {len(rows):>2}. {r['name']:<18} {r['model'].split('/')[-1]}  " + ui.c("dim", host))
    try:
        ans = input(ui.c("dim", "  Number to switch (Enter to keep the current model): ")).strip()
    except EOFError:
        return
    if not ans.isdigit() or not 1 <= int(ans) <= len(rows):
        return
    kind, ref = rows[int(ans) - 1]
    try:
        if kind == "remote":
            from ..remote import probe as check
            check(ref["base_url"], ref.get("api_key"))
            new = remote_backend(ref)
        else:
            if not probe.local_path(ref):
                size = next((m["size_gb"] for m in MODELS if m["id"] == ref), "?")
                if ui.ask(f"Download {ref} ({size} GB) now?") not in ("y", "a"):
                    return
            close = getattr(agent.be, "close", None)
            if close:
                close()   # free memory before loading another local model
            new = LocalBackend(ref, ui.info)
    except Exception as e:
        ui.error(str(e))
        return
    agent.be = new
    agent.budget = min(int(agent.cfg["context"]) if kind == "local" else max(int(agent.cfg["context"]), 16384),
                       new.max_context())
    ui.info(f"Now using {new.name} · {agent.strategy()} mode · context {agent.budget:,} tokens")


def repl(agent: Agent) -> None:
    ui = agent.ui
    try:
        import readline  # noqa: F401  (line editing + history)
        hist = C.USER_DIR / "history"
        hist.parent.mkdir(parents=True, exist_ok=True)
        try:
            readline.read_history_file(hist)
        except OSError:
            pass
        import atexit
        atexit.register(lambda: readline.write_history_file(hist))
    except ImportError:
        pass
    while True:
        try:
            line = input(ui.c("accent", "\n› ") if ui.color else "\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            ui.out("")
            break
        if not line:
            continue
        if line.startswith("/"):
            if not slash(agent, line):
                break
            continue
        try:
            agent.ask(line)
        except KeyboardInterrupt:
            agent.be.stop()
            ui.info("stopped")
    agent.mcp.close()


def main(args) -> int:
    root = Path(args.dir or ".").resolve()
    cfg = C.load(root, {"model": args.model, "base_url": args.base_url, "api_key": args.api_key,
                        "context": args.context, "theme": args.theme,
                        "mode": "yolo" if args.yolo else ("plan" if args.plan else ("auto" if args.auto else None))})
    if getattr(args, "strategy", None):
        cfg["strategy"] = args.strategy
    if getattr(args, "remote", None):
        cfg["remote"] = args.remote
    ui = UI(cfg["theme"])
    from .backend import LocalBackend, RemoteBackend, find_remote, remote_backend
    if cfg.get("remote"):
        r = find_remote(cfg["remote"])
        if not r:
            ui.error(f"No saved connection named {cfg['remote']!r}. Add one in ablate chat → Models → Self-hosted.")
            return 2
        be = remote_backend(r)
        cfg["context"] = max(int(cfg["context"]), 16384) if args.context is None else cfg["context"]
    elif cfg.get("base_url"):
        be = RemoteBackend(cfg["base_url"], cfg["model"], cfg.get("api_key"))
    else:
        be = LocalBackend(cfg["model"], ui.info)
    agent = Agent(root, cfg, ui, be)
    if args.resume and agent.resume():
        ui.info(f"Resumed the last session ({len(agent.history)} messages).")
    if not args.prompt and getattr(args, "prompt_words", None):
        args.prompt = " ".join(args.prompt_words)
    if args.prompt:
        final = agent.ask(args.prompt)
        agent.mcp.close()
        return 0 if final else 1
    extras = []
    if agent.test_cmd:
        extras.append(f"tests   {agent.test_cmd}")
    if agent.skills:
        extras.append(f"skills  {', '.join(list(agent.skills)[:6])}")
    if agent.mcp:
        extras.append(f"mcp     {', '.join(agent.mcp.names())}")
    ui.banner(f"{be.name} · {be.device}", str(root), f"{cfg['mode']} · {agent.strategy()} mode", agent.budget, extras)
    repl(agent)
    return 0
