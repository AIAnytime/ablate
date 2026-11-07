"""Skills: reusable instructions the agent loads only when it needs them.

A skill is a folder with a ``SKILL.md`` whose front matter has ``name`` and
``description``. Only the one-line descriptions go into the prompt; the body is
loaded through the ``skill`` tool. Looked up in, first match wins:

  <project>/.ablate/skills/   ~/.ablate/code/skills/   <project>/.claude/skills/
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from .config import USER_DIR


def _front(text: str) -> Dict[str, str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    out: Dict[str, str] = {}
    if m:
        for line in m.group(1).splitlines():
            k, _, v = line.partition(":")
            if v:
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def find(root: Path) -> Dict[str, dict]:
    skills: Dict[str, dict] = {}
    for base in (root / ".ablate" / "skills", USER_DIR / "skills", root / ".claude" / "skills"):
        if not base.is_dir():
            continue
        for md in sorted(base.glob("*/SKILL.md")):
            text = md.read_text(encoding="utf-8", errors="replace")
            meta = _front(text)
            name = meta.get("name") or md.parent.name
            if name not in skills:
                skills[name] = {"name": name, "description": meta.get("description", "")[:160], "path": md}
    return skills


def listing(skills: Dict[str, dict], limit: int = 12) -> str:
    rows: List[str] = [f"- {s['name']}: {s['description']}" for s in list(skills.values())[:limit]]
    return "\n".join(rows)


def load(skills: Dict[str, dict], name: str, max_chars: int = 5000) -> str:
    s = skills.get(name)
    if not s:
        return f"No skill named {name!r}. Available: {', '.join(skills) or 'none'}."
    body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", s["path"].read_text(encoding="utf-8", errors="replace"), flags=re.S)
    return body[:max_chars] + ("\n… (skill truncated)" if len(body) > max_chars else "")
