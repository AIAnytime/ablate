"""Local storage for ``ablate chat``: profile, chats, memories, settings.

Everything lives in one SQLite file, ``~/.ablate/chat/ablate.db`` (override the
folder with ``ABLATE_HOME``). SQLite ships with Python, so there is no database
server to install and nothing leaves the machine. Deleting the folder resets the
app. Data from the earlier JSON-file layout is imported automatically once.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

HOME = Path(os.environ.get("ABLATE_HOME") or (Path.home() / ".ablate")) / "chat"
DB = HOME / "ablate.db"
_lock = threading.RLock()
_ready = False

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
    "theme": "dark",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS profile (
  id INTEGER PRIMARY KEY CHECK (id = 1), name TEXT NOT NULL, salt TEXT NOT NULL, pass_hash TEXT NOT NULL DEFAULT '',
  created REAL, accepted_terms_at REAL, api_key TEXT, avatar TEXT);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, text TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
  position INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS chats (id TEXT PRIMARY KEY, title TEXT NOT NULL, created REAL, updated REAL, opened REAL,
  model TEXT, pinned INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS messages (chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
  idx INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, ts REAL, model TEXT, steps TEXT, stats TEXT,
  PRIMARY KEY (chat_id, idx));
CREATE INDEX IF NOT EXISTS chats_updated ON chats(updated);
"""


def _connect() -> sqlite3.Connection:
    HOME.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    return con


def _db():
    """Context manager: one connection per operation, serialised by a lock."""
    class _Ctx:
        def __enter__(self):
            _ensure()
            _lock.acquire()
            self.con = _connect()
            return self.con

        def __exit__(self, exc_type, *_):
            try:
                if exc_type is None:
                    self.con.commit()
                self.con.close()
            finally:
                _lock.release()
    return _Ctx()


def _ensure() -> None:
    """Create the database (and import any 0.4.0 JSON data) on first use, not at import time."""
    global _ready
    if _ready:
        return
    with _lock:
        if _ready:
            return
        _ready = True
        try:
            _init()
        except BaseException:
            _ready = False
            raise


def _init() -> None:
    fresh = not DB.exists()
    with _db() as con:
        con.executescript(SCHEMA)
    try:
        os.chmod(DB, 0o600)
    except OSError:
        pass
    if fresh:
        _import_json()


def _import_json() -> None:
    """One-time import from the 0.4.0 layout (profile.json, chats/*.json, ...). Leaves the files in place."""
    def read(p: Path, default):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default
    p = read(HOME / "profile.json", None)
    with _db() as con:
        if p:
            con.execute("INSERT OR REPLACE INTO profile (id, name, salt, pass_hash, created, accepted_terms_at, api_key)"
                        " VALUES (1, ?, ?, ?, ?, ?, ?)", (p.get("name", "You"), p.get("salt", secrets.token_hex(16)),
                                                          p.get("pass_hash", ""), p.get("created"),
                                                          p.get("accepted_terms_at"), p.get("api_key")))
        for k, v in read(HOME / "settings.json", {}).items():
            con.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (k, json.dumps(v)))
        for i, m in enumerate(read(HOME / "memory.json", [])):
            con.execute("INSERT OR REPLACE INTO memories VALUES (?, ?, ?, ?)",
                        (m.get("id") or uuid.uuid4().hex[:10], m.get("text", ""), int(m.get("on", True)), i))
    if (HOME / "chats").is_dir():
        for f in (HOME / "chats").glob("*.json"):
            c = read(f, None)
            if c and c.get("id"):
                save_chat(c, touch=False)


# -- profile (local sign-in) ------------------------------------------------ #
def _hash(passcode: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", passcode.encode(), bytes.fromhex(salt), 200_000).hex()


def profile() -> Optional[dict]:
    with _db() as con:
        r = con.execute("SELECT * FROM profile WHERE id = 1").fetchone()
    return dict(r) if r else None


def public_profile() -> Optional[dict]:
    p = profile()
    if not p:
        return None
    return {"name": p["name"], "has_passcode": bool(p["pass_hash"]), "accepted_terms": bool(p["accepted_terms_at"]),
            "api_key": p["api_key"], "avatar": p["avatar"]}


def create_profile(name: str, passcode: str = "") -> dict:
    salt = secrets.token_hex(16)
    with _db() as con:
        con.execute("INSERT OR REPLACE INTO profile (id, name, salt, pass_hash, created, api_key) VALUES (1, ?, ?, ?, ?, ?)",
                    ((name or "You").strip()[:40], salt, _hash(passcode, salt) if passcode else "", time.time(),
                     "abl-local-" + secrets.token_urlsafe(24)))
    return profile()


def check_passcode(passcode: str) -> bool:
    p = profile()
    if not p:
        return False
    if not p["pass_hash"]:
        return True
    return hmac.compare_digest(_hash(passcode or "", p["salt"]), p["pass_hash"])


def accept_terms() -> None:
    with _db() as con:
        con.execute("UPDATE profile SET accepted_terms_at = ? WHERE id = 1", (time.time(),))


def update_profile(name: Optional[str] = None, passcode: Optional[str] = None, avatar: Optional[str] = None) -> None:
    p = profile()
    if not p:
        return
    with _db() as con:
        if name is not None and name.strip():
            con.execute("UPDATE profile SET name = ? WHERE id = 1", (name.strip()[:40],))
        if passcode is not None:
            con.execute("UPDATE profile SET pass_hash = ? WHERE id = 1", (_hash(passcode, p["salt"]) if passcode else "",))
        if avatar is not None:
            con.execute("UPDATE profile SET avatar = ? WHERE id = 1", (avatar or None,))


# -- settings + memory -------------------------------------------------------- #
def settings() -> dict:
    with _db() as con:
        rows = con.execute("SELECT key, value FROM settings").fetchall()
    return {**DEFAULT_SETTINGS, **{r["key"]: json.loads(r["value"]) for r in rows if r["key"] in DEFAULT_SETTINGS}}


def save_settings(new: dict) -> dict:
    with _db() as con:
        for k, v in new.items():
            if k in DEFAULT_SETTINGS:
                con.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (k, json.dumps(v)))
    return settings()


def memories() -> List[dict]:
    with _db() as con:
        rows = con.execute("SELECT id, text, enabled FROM memories ORDER BY position").fetchall()
    return [{"id": r["id"], "text": r["text"], "on": bool(r["enabled"])} for r in rows]


def save_memories(items: List[dict]) -> List[dict]:
    clean = [{"id": m.get("id") or uuid.uuid4().hex[:10], "text": str(m.get("text", "")).strip()[:500],
              "on": bool(m.get("on", True))} for m in items if str(m.get("text", "")).strip()]
    with _db() as con:
        con.execute("DELETE FROM memories")
        con.executemany("INSERT INTO memories VALUES (?, ?, ?, ?)",
                        [(m["id"], m["text"], int(m["on"]), i) for i, m in enumerate(clean)])
    return clean


# -- chats -------------------------------------------------------------------- #
def list_chats() -> List[dict]:
    with _db() as con:
        rows = con.execute("SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS n FROM chats c "
                           "ORDER BY c.pinned DESC, c.updated DESC").fetchall()
    return [{"id": r["id"], "title": r["title"], "updated": r["updated"], "opened": r["opened"],
             "pinned": bool(r["pinned"]), "model": r["model"], "n": r["n"]} for r in rows]


def get_chat(chat_id: str, mark_opened: bool = False) -> Optional[dict]:
    with _db() as con:
        c = con.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if not c:
            return None
        if mark_opened:
            con.execute("UPDATE chats SET opened = ? WHERE id = ?", (time.time(), chat_id))
        msgs = con.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY idx", (chat_id,)).fetchall()
    out = {k: c[k] for k in ("id", "title", "created", "updated", "opened", "model")}
    out["pinned"] = bool(c["pinned"])
    out["messages"] = []
    for m in msgs:
        d = {"role": m["role"], "content": m["content"], "ts": m["ts"]}
        if m["model"]:
            d["model"] = m["model"]
        if m["steps"]:
            d["steps"] = json.loads(m["steps"])
        if m["stats"]:
            d["stats"] = json.loads(m["stats"])
        out["messages"].append(d)
    return out


def new_chat(model: Optional[str] = None) -> dict:
    now = time.time()
    return {"id": uuid.uuid4().hex[:12], "title": "New chat", "created": now, "updated": now, "opened": now,
            "model": model, "pinned": False, "messages": []}


def save_chat(chat: dict, touch: bool = True) -> None:
    if touch:
        chat["updated"] = time.time()
    with _db() as con:
        con.execute("INSERT INTO chats (id, title, created, updated, opened, model, pinned) VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET title=excluded.title, updated=excluded.updated, model=excluded.model, "
                    "pinned=excluded.pinned, opened=COALESCE(excluded.opened, chats.opened)",
                    (chat["id"], chat.get("title", "New chat"), chat.get("created"), chat.get("updated"),
                     chat.get("opened"), chat.get("model"), int(bool(chat.get("pinned")))))
        con.execute("DELETE FROM messages WHERE chat_id = ?", (chat["id"],))
        con.executemany("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
            (chat["id"], i, m["role"], m.get("content", ""), m.get("ts"), m.get("model"),
             json.dumps(m["steps"]) if m.get("steps") else None, json.dumps(m["stats"]) if m.get("stats") else None)
            for i, m in enumerate(chat.get("messages", []))])


def update_chat(chat_id: str, **fields) -> None:
    with _db() as con:
        if "title" in fields:
            con.execute("UPDATE chats SET title = ? WHERE id = ?", (str(fields["title"])[:120], chat_id))
        if "pinned" in fields:
            con.execute("UPDATE chats SET pinned = ? WHERE id = ?", (int(bool(fields["pinned"])), chat_id))


def delete_chat(chat_id: str) -> None:
    with _db() as con:
        con.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def search_chats(query: str, limit: int = 20) -> List[dict]:
    """Search chat titles and message text (case-insensitive). Newest first, one hit per chat."""
    q = query.strip()
    if not q:
        return []
    like = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    with _db() as con:
        rows = con.execute(
            "SELECT c.id, c.title, c.updated, "
            " (SELECT m.content FROM messages m WHERE m.chat_id = c.id AND m.content LIKE ? ESCAPE '\\' ORDER BY m.idx LIMIT 1) AS hit, "
            " (SELECT m.role FROM messages m WHERE m.chat_id = c.id AND m.content LIKE ? ESCAPE '\\' ORDER BY m.idx LIMIT 1) AS role "
            "FROM chats c WHERE c.title LIKE ? ESCAPE '\\' OR EXISTS "
            " (SELECT 1 FROM messages m WHERE m.chat_id = c.id AND m.content LIKE ? ESCAPE '\\') "
            "ORDER BY c.updated DESC LIMIT ?", (like, like, like, like, limit)).fetchall()
    out = []
    for r in rows:
        text = r["hit"] or ""
        i = text.lower().find(q.lower())
        snippet = (("…" if i > 60 else "") + text[max(0, i - 60): i + 140].replace("\n", " ")) if i >= 0 else ""
        out.append({"id": r["id"], "chat": r["title"], "title": r["title"], "role": r["role"], "snippet": snippet,
                    "updated": r["updated"]})
    return out

