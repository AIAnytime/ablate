"""Workspace tools for ``ablate code`` and the permission rules around them.

Small models make small mistakes (an off-by-one indent, a path typo), so every
tool returns an error message the model can act on rather than a stack trace,
and every result is truncated to fit a small context window.
"""
from __future__ import annotations

import difflib
import fnmatch
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__", ".mypy_cache",
             ".pytest_cache", ".ruff_cache", "dist", "build", ".next", ".nuxt", "target", ".idea", ".vscode",
             ".tox", ".cache", "coverage", ".ablate", ".gradle", "vendor"}
TEXT_LIMIT = 6000          # characters of tool output sent back to the model
READ_WINDOW = 150          # default lines per read_file call

# Commands that are never auto-approved, even in yolo mode.
DANGEROUS = [r"\brm\s+-[a-z]*r[a-z]*f?\s+(/|~|\$HOME)(\s|$)", r"\bmkfs\b", r"\bdd\b.*\bof=/dev/", r":\(\)\s*\{",
             r"\bshutdown\b", r"\breboot\b", r"\bchmod\s+-R\s+777\s+/", r">\s*/dev/sd", r"\bgit\s+push\b.*--force"]


def clip(text: str, limit: int = TEXT_LIMIT) -> str:
    """Keep the head and tail of long output; the middle is usually least useful."""
    if len(text) <= limit:
        return text
    head, tail = text[: limit * 2 // 5], text[-limit * 3 // 5:]
    return f"{head}\n… [{len(text) - limit:,} characters cut] …\n{tail}"


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.undo: List[Tuple[Path, Optional[str]]] = []   # (path, previous content or None if new)
        self.log: List[str] = []                           # short record of actions, used for compaction
        self.scratch = self.root / ".ablate" / "scratchpad.md"
        self.test_cmd: Optional[str] = None

    # -- paths ---------------------------------------------------------------- #
    def path(self, p: str) -> Path:
        q = (self.root / str(p).strip().lstrip("@")).resolve()
        if q != self.root and self.root not in q.parents:
            raise ValueError(f"{p} is outside the project folder")
        return q

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.root)) if p != self.root else "."

    def files(self, limit: int = 4000) -> List[str]:
        out: List[str] = []
        try:
            r = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=self.root, capture_output=True,
                               text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                out = [ln for ln in r.stdout.splitlines() if not any(part in SKIP_DIRS for part in ln.split("/"))]
                return out[:limit]
        except (OSError, subprocess.SubprocessError):
            pass
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for f in sorted(filenames):
                out.append(self.rel(Path(dirpath) / f))
                if len(out) >= limit:
                    return out
        return out

    # -- repo understanding (done by the harness, not the model) ---------------- #
    def repo_map(self, max_chars: int = 1600) -> str:
        files = self.files()
        if not files:
            return "(empty folder)"
        lines, used = [], 0
        dirs: Dict[str, List[str]] = {}
        for f in files:
            d, _, name = f.rpartition("/")
            dirs.setdefault(d or ".", []).append(name)
        for d in sorted(dirs, key=lambda x: (x.count("/"), x)):
            names = dirs[d]
            row = f"{d}/: " + ", ".join(names[:12]) + (f", … +{len(names) - 12}" if len(names) > 12 else "")
            if used + len(row) > max_chars:
                lines.append(f"… {len(dirs) - len(lines)} more folders")
                break
            lines.append(row)
            used += len(row)
        return f"{len(files)} files\n" + "\n".join(lines)

    def detect_tests(self) -> Optional[str]:
        r = self.root
        if (r / "package.json").exists():
            try:
                if json.loads((r / "package.json").read_text()).get("scripts", {}).get("test"):
                    return "npm test"
            except (OSError, ValueError):
                pass
        if (r / "Cargo.toml").exists():
            return "cargo test"
        if (r / "go.mod").exists():
            return "go test ./..."
        if (r / "pytest.ini").exists() or (r / "tests").is_dir() or (r / "test").is_dir() or \
                ("pytest" in (r / "pyproject.toml").read_text(errors="ignore") if (r / "pyproject.toml").exists() else False):
            return "python -m pytest -q"
        return None

    def instructions(self, max_chars: int = 2400) -> str:
        for name in ("ABLATE.md", "AGENTS.md", "CLAUDE.md", ".ablate/ABLATE.md"):
            p = self.root / name
            if p.is_file():
                return f"From {name}:\n" + clip(p.read_text(errors="replace"), max_chars)
        return ""

    def relevant(self, task: str, max_hits: int = 6) -> str:
        """Grep for identifiers and file names mentioned in the request, so a small model starts in the right place."""
        words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9]+)?", task))
        keys = [w for w in words if (("_" in w or any(c.isupper() for c in w[1:]) or "." in w) and len(w) > 3)]
        hits: List[str] = []
        files = self.files()
        stop = {"this", "that", "with", "when", "file", "files", "code", "test", "tests", "make", "should", "from",
                "into", "what", "where", "which", "there", "their", "check", "please", "about", "after", "crash", "error"}
        for w in sorted({w.lower() for w in words if len(w) >= 4 and w.lower() not in stop}, key=len, reverse=True):
            for f in files:
                stem = f.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
                if (w == stem or w in stem.split("_")) and not any(f in h for h in hits):
                    hits.append(f"{f} (file name matches '{w}')")
        for k in sorted(keys, key=len, reverse=True)[:5]:
            for f in files:
                if k.lower() in f.lower():
                    hits.append(f"{f} (file name matches '{k}')")
            if len(hits) >= max_hits:
                break
            for line in self.grep(re.escape(k), max_hits=3).splitlines():
                if ":" in line and line not in hits:
                    hits.append(line)
        return "\n".join(hits[:max_hits])

    def preload(self, hits: str, max_lines: int = 160) -> str:
        """Numbered contents of the best-matching source file, if it is short. Saves a small model several steps."""
        for line in hits.splitlines():
            f = line.split(" (file name", 1)[0].split(":", 1)[0]
            if "test" in f.lower():
                continue
            p = self.root / f
            if p.is_file() and p.suffix not in (".json", ".lock", ".md", ".txt"):
                try:
                    text = p.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                lines = text.splitlines()
                if 0 < len(lines) <= max_lines:
                    self.log.append(f"read {f} 1-{len(lines)} (preloaded)")
                    return f"Contents of {f}:\n" + "\n".join(f"{i:>5}  {ln}" for i, ln in enumerate(lines, 1))
        return ""

    # -- tools ------------------------------------------------------------------- #
    def list_files(self, path: str = ".", pattern: str = "*") -> str:
        base = self.path(path)
        prefix = "" if base == self.root else self.rel(base) + "/"
        matches = [f for f in self.files() if f.startswith(prefix) and fnmatch.fnmatch(f.rsplit("/", 1)[-1], pattern or "*")]
        if not matches:
            return f"No files match {pattern!r} under {path}."
        return f"{len(matches)} files" + ("" if len(matches) <= 200 else " (first 200)") + ":\n" + "\n".join(matches[:200])

    def grep(self, pattern: str, path: str = ".", glob: str = "", max_hits: int = 60) -> str:
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))
        base = self.path(path)
        prefix = "" if base == self.root else self.rel(base)
        out: List[str] = []
        for f in self.files():
            if prefix and not (f == prefix or f.startswith(prefix + "/")):
                continue
            if glob and not fnmatch.fnmatch(f.rsplit("/", 1)[-1], glob):
                continue
            p = self.root / f
            try:
                if p.stat().st_size > 2_000_000:
                    continue
                with open(p, encoding="utf-8", errors="strict") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            out.append(f"{f}:{i}: {line.rstrip()[:200]}")
                            if len(out) >= max_hits:
                                return "\n".join(out) + f"\n(stopped at {max_hits} matches; narrow the pattern or path)"
            except (UnicodeDecodeError, OSError):
                continue
        return "\n".join(out) or f"No matches for {pattern!r}."

    def read_file(self, path: str, start: int = 1, end: Optional[int] = None) -> str:
        p = self.path(path)
        if p.is_dir():
            return self.list_files(path)
        if not p.exists():
            close = difflib.get_close_matches(str(path), self.files(), n=3, cutoff=0.5)
            return f"Error: {path} does not exist." + (f" Did you mean: {', '.join(close)}?" if close else "")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        try:
            start = max(1, int(start or 1))
        except (TypeError, ValueError):
            start = 1
        try:
            end = int(end) if end else 0
        except (TypeError, ValueError):
            end = 0
        if start > len(lines):
            start = max(1, len(lines) - READ_WINDOW + 1)
        if end < start:  # missing, negative or nonsense: read a window from start
            end = start + READ_WINDOW - 1
        end = min(len(lines), end)
        body = "\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(start, end + 1))
        more = f"\n(lines {start}-{end} of {len(lines)}; call read_file with start={end + 1} for more)" if end < len(lines) else ""
        self.log.append(f"read {self.rel(p)} {start}-{end}")
        return clip(body) + more

    def _match(self, text: str, old: str) -> Tuple[Optional[int], Optional[int], str]:
        """Find ``old`` in ``text``: exact, then ignoring indentation and trailing spaces."""
        n = text.count(old)
        if n == 1:
            i = text.index(old)
            return i, i + len(old), "exact"
        if n > 1:
            return None, None, f"matches {n} places; include more surrounding lines so it is unique"
        tl, ol = text.splitlines(keepends=True), [x.strip() for x in old.strip("\n").splitlines()]
        if not ol:
            return None, None, "old text is empty"
        found = [i for i in range(len(tl) - len(ol) + 1) if [x.strip() for x in tl[i:i + len(ol)]] == ol]
        if len(found) == 1:
            i = found[0]
            a = sum(len(x) for x in tl[:i])
            return a, a + sum(len(x) for x in tl[i:i + len(ol)]), "whitespace-insensitive"
        if len(found) > 1:
            return None, None, f"matches {len(found)} places; include more surrounding lines so it is unique"
        return None, None, "not found"

    def edit_file(self, path: str, old: str, new: str) -> Tuple[str, Optional[Tuple[str, str]]]:
        p = self.path(path)
        if not p.exists():
            return f"Error: {path} does not exist. Use write_file to create it.", None
        text = p.read_text(encoding="utf-8")
        a, b, how = self._match(text, old)
        if a is None:
            lines = text.splitlines()
            first = (old.strip().splitlines() or [""])[0].strip()
            best = difflib.get_close_matches(first, [ln.strip() for ln in lines], n=1, cutoff=0.4)
            hint = ""
            if best:
                k = [ln.strip() for ln in lines].index(best[0])
                hint = "\nClosest lines in the file:\n" + "\n".join(f"{i + 1:>5}  {lines[i]}" for i in range(max(0, k - 3), min(len(lines), k + 6)))
            return f"Error: old text {how} in {path}. Copy it exactly from read_file output (without line numbers).{hint}", None
        if how != "exact":  # keep the file's own indentation for the first line of the replacement
            indent = re.match(r"\s*", text[a:b]).group(0)
            new = indent + new.lstrip() if new.strip() else new
            if text[a:b].endswith("\n") and not new.endswith("\n"):
                new += "\n"
        updated = text[:a] + new + text[b:]
        self.undo.append((p, text))
        p.write_text(updated, encoding="utf-8")
        added, removed = new.count("\n") + 1, text[a:b].count("\n") + 1
        self.log.append(f"edited {self.rel(p)} (+{added} -{removed})")
        return f"Edited {path} ({how} match).", (text, updated)

    def write_file(self, path: str, content: str) -> Tuple[str, Optional[Tuple[str, str]]]:
        p = self.path(path)
        old = p.read_text(encoding="utf-8") if p.exists() else None
        p.parent.mkdir(parents=True, exist_ok=True)
        self.undo.append((p, old))
        p.write_text(content, encoding="utf-8")
        self.log.append(f"{'rewrote' if old is not None else 'created'} {self.rel(p)} ({content.count(chr(10)) + 1} lines)")
        return f"{'Overwrote' if old is not None else 'Created'} {path}.", (old or "", content)

    def bash(self, command: str, timeout: int = 120) -> str:
        try:
            r = subprocess.run(command, shell=True, cwd=self.root, capture_output=True, text=True,
                               timeout=min(int(timeout or 120), 600), stdin=subprocess.DEVNULL)
            out = (r.stdout + ("\n" + r.stderr if r.stderr else "")).strip()
            code = r.returncode
        except subprocess.TimeoutExpired as e:
            out, code = f"{(e.stdout or '')}\n{(e.stderr or '')}\n(timed out after {timeout}s)", -1
        self.log.append(f"ran `{command[:80]}` → exit {code}")
        hint = ""
        if code == 127 and self.test_cmd and command.split()[0] in ("pytest", "npm", "cargo", "go", "python", "python3"):
            hint = f"\nHint: this project's test command is: {self.test_cmd}"
        return f"exit code {code}\n" + clip(out or "(no output)") + hint

    def note(self, text: str) -> str:
        self.scratch.parent.mkdir(parents=True, exist_ok=True)
        with open(self.scratch, "a", encoding="utf-8") as fh:
            fh.write(f"- {text.strip()}\n")
        return "Saved to the scratchpad. It is shown to you at the start of every step."

    def scratchpad(self, max_chars: int = 1200) -> str:
        if not self.scratch.exists():
            return ""
        s = self.scratch.read_text(encoding="utf-8", errors="replace").strip()
        return s[-max_chars:]

    def revert(self) -> str:
        if not self.undo:
            return "Nothing to undo."
        p, prev = self.undo.pop()
        if prev is None:
            p.unlink(missing_ok=True)
            return f"Removed {self.rel(p)} (it was created by the agent)."
        p.write_text(prev, encoding="utf-8")
        return f"Restored {self.rel(p)}."


# --------------------------------------------------------------------------- #
# Schemas: short descriptions on purpose; every token here is paid on every step.
# --------------------------------------------------------------------------- #
def _fn(name: str, desc: str, props: dict, required: List[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


S = {"type": "string"}
I = {"type": "integer"}
SCHEMAS = {
    "list_files": _fn("list_files", "List project files under a folder, optionally filtered by a glob like '*.py'.",
                      {"path": S, "pattern": S}, []),
    "grep": _fn("grep", "Search file contents with a regex. Returns file:line: text.",
                {"pattern": S, "path": S, "glob": S}, ["pattern"]),
    "read_file": _fn("read_file", "Read a file with line numbers. Use start/end for long files.",
                     {"path": S, "start": I, "end": I}, ["path"]),
    "edit_file": _fn("edit_file", "Replace one exact snippet in a file. 'old' must appear once; copy it from read_file without line numbers.",
                     {"path": S, "old": S, "new": S}, ["path", "old", "new"]),
    "write_file": _fn("write_file", "Create a new file or fully replace a small file.", {"path": S, "content": S}, ["path", "content"]),
    "bash": _fn("bash", "Run a shell command in the project folder (tests, git, build). Returns exit code and output.",
                {"command": S}, ["command"]),
    "note": _fn("note", "Save a short note (plan, finding, next step) to your scratchpad so it survives a full context.",
                {"text": S}, ["text"]),
    "skill": _fn("skill", "Load the full instructions of a skill listed in the system prompt.", {"name": S}, ["name"]),
    "mcp": _fn("mcp", "Use a tool from an MCP server. Call with only 'server' to list its tools; then call with "
                      "'tool' and 'arguments'.", {"server": S, "tool": S, "arguments": {"type": "object"}}, ["server"]),
}
READ_ONLY = {"list_files", "grep", "read_file", "note", "skill"}


def command_allowed(cmd: str, allow: List[str], deny: List[str]) -> Optional[bool]:
    """True = allowlisted, False = denied, None = ask."""
    c = cmd.strip()
    if any(re.search(rx, c) for rx in DANGEROUS):
        return False if any(fnmatch.fnmatch(c, d) for d in deny) else None
    if any(fnmatch.fnmatch(c, d) for d in deny):
        return False
    if re.search(r"[;&|`$<>]", c):  # chained or redirected commands always ask
        return None
    return True if any(fnmatch.fnmatch(c, a) for a in allow) else None


def is_dangerous(cmd: str) -> bool:
    return any(re.search(rx, cmd.strip()) for rx in DANGEROUS)


def have(binary: str) -> bool:
    return shutil.which(binary) is not None
