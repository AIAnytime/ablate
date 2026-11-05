"""Local storage for ``ablate chat``: profile, chats, memories, settings.

Everything is plain JSON under ``~/.ablate/chat`` (override with ``ABLATE_HOME``).
Nothing is sent anywhere; deleting that folder resets the app.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

HOME = Path(os.environ.get("ABLATE_HOME") or (Path.home() / ".ablate")) / "chat"
CHATS = HOME / "chats"
_lock = threading.RLock()

DEFAULT_SETTINGS = {
    "system": ("You are Ablate, a helpful and direct assistant running privately on the user's own "
               "computer. Answer the question that was asked, fully and plainly."),
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 768,
    "context_tokens": 4096,
    "repetition_penalty": 1.1,
    "use_memory": True,
    "tools": True,
    "network_tools": False,
}


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


# -- profile (local sign-in) ------------------------------------------------ #
def _hash(passcode: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", passcode.encode(), bytes.fromhex(salt), 200_000).hex()


def profile() -> Optional[dict]:
    return _read(HOME / "profile.json", None)


def public_profile() -> Optional[dict]:
    p = profile()
    if not p:
        return None
    return {"name": p["name"], "has_passcode": bool(p.get("pass_hash")),
            "accepted_terms": bool(p.get("accepted_terms_at")), "api_key": p.get("api_key")}


def create_profile(name: str, passcode: str = "") -> dict:
    with _lock:
        salt = secrets.token_hex(16)
        p = {"name": (name or "You").strip()[:40], "salt": salt,
             "pass_hash": _hash(passcode, salt) if passcode else "",
             "created": time.time(), "accepted_terms_at": None,
             "api_key": "abl-local-" + secrets.token_urlsafe(24)}
        _write(HOME / "profile.json", p)
        return p


def check_passcode(passcode: str) -> bool:
    p = profile()
    if not p:
        return False
    if not p.get("pass_hash"):
        return True
    return hmac.compare_digest(_hash(passcode or "", p["salt"]), p["pass_hash"])


def accept_terms() -> None:
    with _lock:
        p = profile()
        if p:
            p["accepted_terms_at"] = time.time()
            _write(HOME / "profile.json", p)


def update_profile(name: Optional[str] = None, passcode: Optional[str] = None) -> None:
    with _lock:
        p = profile()
        if not p:
            return
        if name is not None:
            p["name"] = name.strip()[:40] or p["name"]
        if passcode is not None:
            p["pass_hash"] = _hash(passcode, p["salt"]) if passcode else ""
        _write(HOME / "profile.json", p)


# -- settings + memory -------------------------------------------------------- #
def settings() -> dict:
    return {**DEFAULT_SETTINGS, **_read(HOME / "settings.json", {})}


def save_settings(new: dict) -> dict:
    cur = settings()
    cur.update({k: v for k, v in new.items() if k in DEFAULT_SETTINGS})
    _write(HOME / "settings.json", cur)
    return cur


def memories() -> List[dict]:
    return _read(HOME / "memory.json", [])


def save_memories(items: List[dict]) -> List[dict]:
    clean = [{"id": m.get("id") or uuid.uuid4().hex[:10], "text": str(m.get("text", "")).strip()[:500],
              "on": bool(m.get("on", True))} for m in items if str(m.get("text", "")).strip()]
    _write(HOME / "memory.json", clean)
    return clean


# -- chats -------------------------------------------------------------------- #
def _chat_path(chat_id: str) -> Path:
    safe = "".join(c for c in chat_id if c.isalnum() or c in "-_")[:40]
    return CHATS / f"{safe}.json"


def list_chats() -> List[dict]:
    out = []
    if CHATS.is_dir():
        for f in CHATS.glob("*.json"):
            c = _read(f, None)
            if c:
                out.append({"id": c["id"], "title": c.get("title", "New chat"), "updated": c.get("updated", 0),
                            "pinned": c.get("pinned", False), "model": c.get("model"),
                            "n": len(c.get("messages", []))})
    return sorted(out, key=lambda c: (not c["pinned"], -c["updated"]))


def get_chat(chat_id: str) -> Optional[dict]:
    return _read(_chat_path(chat_id), None)


def new_chat(model: Optional[str] = None) -> dict:
    now = time.time()
    return {"id": uuid.uuid4().hex[:12], "title": "New chat", "created": now, "updated": now,
            "model": model, "pinned": False, "messages": []}


def save_chat(chat: dict) -> None:
    chat["updated"] = time.time()
    with _lock:
        _write(_chat_path(chat["id"]), chat)


def delete_chat(chat_id: str) -> None:
    try:
        _chat_path(chat_id).unlink()
    except OSError:
        pass


def search_chats(query: str, limit: int = 5) -> List[dict]:
    """Case-insensitive substring search across saved chats (used by the search tool)."""
    q = query.lower().strip()
    hits = []
    if not q or not CHATS.is_dir():
        return hits
    for f in CHATS.glob("*.json"):
        c = _read(f, None) or {}
        for m in c.get("messages", []):
            text = m.get("content", "")
            i = text.lower().find(q)
            if i >= 0:
                hits.append({"chat": c.get("title", ""), "role": m.get("role"),
                             "snippet": text[max(0, i - 120): i + 240].replace("\n", " ")})
                break
        if len(hits) >= limit:
            break
    return hits
