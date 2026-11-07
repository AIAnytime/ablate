"""Configuration for ``ablate code``.

Two JSON files are merged, project over user:
  ~/.ablate/code/config.json      (user-wide)
  <project>/.ablate/config.json   (this repository)
CLI flags override both. JSON rather than TOML so it works on Python 3.9.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

USER_DIR = Path(os.environ.get("ABLATE_HOME") or (Path.home() / ".ablate")) / "code"

DEFAULTS: Dict[str, Any] = {
    # model: a HuggingFace id run locally, or any model name on an OpenAI-compatible server (base_url)
    "model": "ai-anytime/qwen-1.5b-abliterated",
    "base_url": None,          # e.g. http://localhost:8000/v1 (vLLM), http://localhost:11434/v1 (Ollama)
    "api_key": None,
    "context": 8192,           # token budget for one request; small models need a tight budget
    "max_new_tokens": 1024,
    "temperature": 0.2,
    "max_steps": 30,           # tool calls per request before the agent must stop
    "strategy": "auto",        # auto | agent (model drives tools) | edit (harness drives, model writes the fix)
    "edit_rounds": 3,          # edit mode: fix -> test -> feedback rounds
    "remote": None,            # name or id of a saved RunPod/server connection (shared with ablate chat)
    "mode": "ask",             # ask | auto (auto-approve edits) | plan (read-only) | yolo (approve everything)
    "theme": "ablate",
    "allow": ["git status*", "git diff*", "git log*", "ls*", "pwd", "cat *", "head *", "tail *", "wc *",
              "pytest*", "python -m pytest*", "npm test*", "cargo test*", "go test*"],
    "deny": [],
    "test_command": None,      # detected automatically when left empty
    "mcp": {"servers": {}},    # name -> {"command": "...", "args": [...], "env": {...}}
}


def _read(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load(root: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    for layer in (_read(USER_DIR / "config.json"), _read(root / ".ablate" / "config.json")):
        for k, v in layer.items():
            if k == "mcp" and isinstance(v, dict):
                servers = dict(cfg["mcp"].get("servers", {}))
                servers.update(v.get("servers", {}))
                cfg["mcp"] = {"servers": servers}
            elif k in ("allow", "deny") and isinstance(v, list):
                cfg[k] = list(dict.fromkeys(cfg[k] + v))
            else:
                cfg[k] = v
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def save_user(updates: Dict[str, Any]) -> Path:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    p = USER_DIR / "config.json"
    data = _read(p)
    data.update(updates)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p
