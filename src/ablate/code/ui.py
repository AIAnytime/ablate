"""Terminal rendering for ``ablate code``: themes, tool lines, diffs, prompts.

Plain ANSI escape codes and the standard library only.
"""
from __future__ import annotations

import difflib
import itertools
import os
import shutil
import sys
import threading
import time
from typing import List, Optional

THEMES = {
    "ablate": {"accent": "38;5;141", "accent2": "38;5;111", "dim": "38;5;244", "ok": "38;5;78", "err": "38;5;203",
               "warn": "38;5;214", "add": "38;5;78", "del": "38;5;203", "code": "38;5;189", "bold": "1"},
    "dark": {"accent": "38;5;75", "accent2": "38;5;117", "dim": "38;5;245", "ok": "38;5;114", "err": "38;5;210",
             "warn": "38;5;221", "add": "38;5;114", "del": "38;5;210", "code": "38;5;252", "bold": "1"},
    "light": {"accent": "38;5;61", "accent2": "38;5;25", "dim": "38;5;242", "ok": "38;5;28", "err": "38;5;160",
              "warn": "38;5;130", "add": "38;5;28", "del": "38;5;160", "code": "38;5;236", "bold": "1"},
    "mono": {k: "" for k in ("accent", "accent2", "dim", "ok", "err", "warn", "add", "del", "code")} | {"bold": "1"},
}


class UI:
    def __init__(self, theme: str = "ablate", color: Optional[bool] = None) -> None:
        self.color = sys.stdout.isatty() and not os.environ.get("NO_COLOR") if color is None else color
        self.set_theme(theme)
        self._spin: Optional[threading.Thread] = None
        self._spinning = threading.Event()

    def set_theme(self, name: str) -> bool:
        if name not in THEMES:
            return False
        self.theme_name, self.t = name, THEMES[name]
        return True

    # -- primitives ---------------------------------------------------------- #
    def c(self, key: str, text: str) -> str:
        code = self.t.get(key, "")
        return f"\033[{code}m{text}\033[0m" if (self.color and code) else text

    def out(self, text: str = "", end: str = "\n") -> None:
        self.stop_spinner()
        sys.stdout.write(text + end)
        sys.stdout.flush()

    def width(self) -> int:
        return min(shutil.get_terminal_size((100, 20)).columns, 120)

    # -- blocks ---------------------------------------------------------------- #
    def banner(self, model: str, root: str, mode: str, ctx: int, extras: List[str]) -> None:
        w = self.width() - 2
        line = "─" * w
        self.out(self.c("accent", "╭" + line + "╮"))
        rows = [self.c("bold", "ablate code") + self.c("dim", "  ·  unrestricted coding agent, running on your machine"),
                "",
                self.c("dim", "model   ") + model,
                self.c("dim", "project ") + root,
                self.c("dim", "mode    ") + mode + self.c("dim", f"   ·   context {ctx:,} tokens")]
        rows += [self.c("dim", e) for e in extras]
        for r in rows:
            self.out(self.c("accent", "│ ") + r)
        self.out(self.c("accent", "╰" + line + "╯"))
        self.out(self.c("dim", "  /help for commands · Ctrl+C stops a reply · Ctrl+D exits"))

    def tool(self, name: str, summary: str) -> None:
        self.out(self.c("accent", "● ") + self.c("bold", name) + self.c("dim", f"({summary})"))

    def result(self, text: str, ok: bool = True, max_lines: int = 6) -> None:
        lines = text.rstrip().splitlines() or ["(no output)"]
        shown = lines[:max_lines]
        for i, ln in enumerate(shown):
            prefix = "  ⎿ " if i == 0 else "    "
            self.out(self.c("dim", prefix) + self.c("dim" if ok else "err", ln[: self.width() - 6]))
        if len(lines) > max_lines:
            self.out(self.c("dim", f"    … {len(lines) - max_lines} more lines"))

    def diff(self, path: str, old: str, new: str, context: int = 2) -> None:
        d = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=context))
        if not d:
            self.out(self.c("dim", "    (no changes)"))
            return
        for ln in d[2:60]:
            if ln.startswith("@@"):
                self.out(self.c("accent2", "    " + ln))
            elif ln.startswith("+"):
                self.out(self.c("add", "    " + ln))
            elif ln.startswith("-"):
                self.out(self.c("del", "    " + ln))
            else:
                self.out(self.c("dim", "    " + ln))
        if len(d) > 62:
            self.out(self.c("dim", f"    … {len(d) - 62} more diff lines"))

    def ask(self, question: str) -> str:
        """Ask a one-letter question. Returns the lowercased answer ('' on EOF)."""
        self.stop_spinner()
        try:
            ans = input(self.c("warn", "  ? ") + question + self.c("dim", "  [y]es / [n]o / [a]lways this session: "))
        except EOFError:
            return ""
        return (ans.strip().lower() or "n")[0]

    def info(self, text: str) -> None:
        self.out(self.c("dim", "  " + text))

    def error(self, text: str) -> None:
        self.out(self.c("err", "  ✗ " + text))

    # -- streaming assistant text ------------------------------------------------ #
    def stream_start(self) -> None:
        self._in_code = False
        self._line = ""

    def stream(self, piece: str) -> None:
        """Print model text as it arrives, colouring fenced code blocks."""
        self.stop_spinner()
        for ch in piece:
            self._line += ch
            if ch == "\n":
                self._flush_line()
        sys.stdout.flush()

    def _flush_line(self) -> None:
        ln = self._line
        self._line = ""
        if ln.strip().startswith("```"):
            self._in_code = not self._in_code
            sys.stdout.write(self.c("dim", ln))
        else:
            sys.stdout.write(self.c("code", ln) if self._in_code else ln)

    def stream_end(self) -> None:
        if self._line:
            self._flush_line()
            sys.stdout.write("\n")
        sys.stdout.flush()

    # -- spinner ----------------------------------------------------------------- #
    def spinner(self, label: str = "thinking") -> None:
        if not self.color or self._spinning.is_set():
            return
        self._spinning.set()

        def run() -> None:
            t0 = time.time()
            for fr in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
                if not self._spinning.is_set():
                    break
                sys.stdout.write("\r" + self.c("accent", fr) + self.c("dim", f" {label}… {time.time() - t0:4.0f}s"))
                sys.stdout.flush()
                time.sleep(0.09)
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()

        self._spin = threading.Thread(target=run, daemon=True)
        self._spin.start()

    def stop_spinner(self) -> None:
        if self._spinning.is_set():
            self._spinning.clear()
            if self._spin:
                self._spin.join(timeout=1)
