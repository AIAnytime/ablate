"""``ablate chat``: a private, local chat app for the AI Anytime unrestricted SLMs.

Python stdlib HTTP server + one HTML file. Models download once from the
HuggingFace Hub, then load from the local cache with no network at all. Chats,
memories and the profile are JSON files under ``~/.ablate/chat``.

Also serves an OpenAI-compatible API at ``/v1/chat/completions`` so editors,
agents and SDKs can talk to the same local model.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import threading
import time
import traceback
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import store, tools

Emit = Callable[[dict], None]

MODELS = [
    {"id": "ai-anytime/qwen-1.5b-abliterated", "name": "Ablate Qwen 1.5B", "params": "1.5B",
     "size_gb": 3.1, "license": "Apache-2.0", "context": 32768, "base": "Qwen/Qwen2.5-1.5B-Instruct",
     "blurb": "Most unrestricted: hard refusals went from 100% to 0%. Good at tools. Recommended.",
     "ram": "8 GB RAM"},
    {"id": "ai-anytime/Llama-3.2-3B-Instruct-abliterated", "name": "Ablate Llama 3B", "params": "3.2B",
     "size_gb": 6.4, "license": "Llama 3.2 Community", "context": 131072, "base": "meta-llama/Llama-3.2-3B-Instruct",
     "blurb": "Most capable writer. Refusals cut from 94% to 59%, so it still declines some asks.",
     "ram": "16 GB RAM"},
    {"id": "ai-anytime/Llama-3.2-1B-Instruct-abliterated", "name": "Ablate Llama 1B", "params": "1.2B",
     "size_gb": 2.5, "license": "Llama 3.2 Community", "context": 131072, "base": "meta-llama/Llama-3.2-1B-Instruct",
     "blurb": "Fastest. Runs on almost any laptop. Refusals cut from 75% to 34%. Too small for tools.",
     "ram": "8 GB RAM", "tools": False},
]
ALLOW = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.md", "tokenizer*"]


# --------------------------------------------------------------------------- #
# Model engine
# --------------------------------------------------------------------------- #
class Engine:
    def __init__(self) -> None:
        self.lock = threading.Lock()          # one generation (or load) at a time
        self.model = None
        self.tok = None
        self.model_id: Optional[str] = None
        self.device = None
        self.dtype = None
        self.stop = threading.Event()
        self.downloads: Dict[str, dict] = {}  # repo -> {done, total, error, finished}

    # -- cache / download ---------------------------------------------------- #
    @staticmethod
    def _cache_dir(repo: str) -> Path:
        from huggingface_hub import constants
        return Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))

    def local_path(self, repo: str) -> Optional[str]:
        """Snapshot folder if the model is fully downloaded, else None."""
        try:
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download(repo, local_files_only=True, allow_patterns=ALLOW))
        except Exception:
            return None
        if not (p / "config.json").exists():
            return None
        idx = p / "model.safetensors.index.json"
        if idx.exists():
            need = set(json.loads(idx.read_text())["weight_map"].values())
            return str(p) if all((p / f).exists() for f in need) else None
        return str(p) if any(p.glob("*.safetensors")) else None

    def _bytes_on_disk(self, repo: str) -> int:
        d = self._cache_dir(repo) / "blobs"
        return sum(f.stat().st_size for f in d.glob("*")) if d.is_dir() else 0

    def download(self, repo: str) -> None:
        if self.downloads.get(repo, {}).get("finished") is False:
            return
        total = int(next((m["size_gb"] for m in MODELS if m["id"] == repo), 0) * 1e9)
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(repo, files_metadata=True)
            total = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors")) or total
        except Exception:
            pass
        st = {"done": self._bytes_on_disk(repo), "total": total, "error": None, "finished": False}
        self.downloads[repo] = st

        def work() -> None:
            try:
                from huggingface_hub import constants, snapshot_download
                from ..ui.server import _find_token
                # plain HTTP writes into the cache as it goes, so the app can show live
                # progress (Xet only writes the file at the end)
                constants.HF_HUB_DISABLE_XET = True
                snapshot_download(repo, allow_patterns=ALLOW, token=_find_token())
            except Exception as e:
                st["error"] = _explain(e)
            st["finished"] = True

        threading.Thread(target=work, daemon=True).start()

    def download_status(self, repo: str) -> Optional[dict]:
        st = self.downloads.get(repo)
        if st and not st["finished"]:
            st["done"] = min(st["total"] or 1 << 62, self._bytes_on_disk(repo))
        return st

    def remove(self, repo: str) -> None:
        if self.model_id == repo:
            self.unload()
        shutil.rmtree(self._cache_dir(repo), ignore_errors=True)
        self.downloads.pop(repo, None)

    # -- load / unload ------------------------------------------------------- #
    def pick(self):
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def load(self, repo: str) -> None:
        if self.model_id == repo:
            return
        path = self.local_path(repo)
        if not path:
            raise RuntimeError("That model is not downloaded yet. Download it first.")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.unload()
        dev, dt = self.pick()
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dt).to(dev).eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self.model, self.tok, self.model_id, self.device, self.dtype = model, tok, repo, dev, dt

    def unload(self) -> None:
        import gc
        self.model = self.tok = self.model_id = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def max_context(self) -> int:
        cfg = getattr(self.model, "config", None)
        return int(getattr(cfg, "max_position_embeddings", 4096) or 4096)

    # -- prompt building ------------------------------------------------------ #
    def encode(self, messages: List[dict], tool_schemas: Optional[List[dict]] = None):
        kw = {"tools": tool_schemas} if tool_schemas else {}
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True, **kw)
        return ids["input_ids"]

    def fit(self, system: str, history: List[dict], limit: int, reserve: int,
            tool_schemas: Optional[List[dict]] = None):
        """Drop the oldest turns until system + history + reply fit in ``limit`` tokens."""
        kept = list(history)
        trimmed = 0
        while True:
            msgs = ([{"role": "system", "content": system}] if system else []) + kept
            ids = self.encode(msgs, tool_schemas)
            if ids.shape[1] + reserve <= limit or len(kept) <= 1:
                return msgs, ids, trimmed
            kept.pop(0)
            trimmed += 1
            while kept and kept[0]["role"] != "user":  # never start mid-exchange
                kept.pop(0)
                trimmed += 1

    # -- generation ---------------------------------------------------------- #
    def stream(self, input_ids, opts: dict, on_text: Callable[[str], None]) -> dict:
        """Generate from ``input_ids``; call ``on_text`` with each new piece. Returns stats."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        stop = self.stop
        stop.clear()

        class _Stop(StoppingCriteria):
            def __call__(self, *a, **k):
                return stop.is_set()

        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True, timeout=300)
        temp = float(opts.get("temperature", 0.7))
        gen = dict(input_ids=input_ids.to(self.device),
                   attention_mask=torch.ones_like(input_ids).to(self.device),
                   max_new_tokens=int(opts.get("max_new_tokens", 768)),
                   repetition_penalty=float(opts.get("repetition_penalty", 1.1)),
                   do_sample=temp > 0, streamer=streamer, pad_token_id=self.tok.pad_token_id,
                   stopping_criteria=StoppingCriteriaList([_Stop()]))
        if temp > 0:
            gen.update(temperature=temp, top_p=float(opts.get("top_p", 0.9)))
        err: List[BaseException] = []

        def work() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(**gen)
            except BaseException as e:  # surface errors from the worker thread
                err.append(e)
                streamer.end()

        t0 = time.time()
        th = threading.Thread(target=work, daemon=True)
        th.start()
        text = ""
        for piece in streamer:
            if piece:
                text += piece
                on_text(piece)
        th.join()
        if err:
            raise err[0]
        n = len(self.tok(text, add_special_tokens=False)["input_ids"]) if text else 0
        secs = max(time.time() - t0, 1e-6)
        return {"text": text, "tokens": n, "seconds": round(secs, 2), "tps": round(n / secs, 1),
                "stopped": stop.is_set()}


E = Engine()
SESSIONS: Dict[str, float] = {}


# --------------------------------------------------------------------------- #
# Chat turn (with tools)
# --------------------------------------------------------------------------- #
def _system_prompt(opts: dict, with_tools: bool) -> str:
    import datetime as _dt
    now = _dt.datetime.now().astimezone()
    parts = [opts.get("system") or "",
             f"Current local date and time: {now:%A, %d %B %Y, %H:%M} ({now.tzname()})."]
    if opts.get("use_memory", True):
        mem = [m["text"] for m in store.memories() if m.get("on")]
        if mem:
            parts.append("Things the user asked you to remember:\n" + "\n".join(f"- {m}" for m in mem))
    if with_tools:
        parts.append("You can call tools when they help (exact arithmetic, the current date or time, the "
                     "user's earlier chats" + (", web pages" if opts.get("network_tools") else "") +
                     "). To call one, reply with only the tool call. Otherwise answer normally.")
    return "\n\n".join(p for p in parts if p.strip())


def chat_turn(body: dict, emit: Emit) -> None:
    if E.model is None:
        raise RuntimeError("Load a model first (top of the screen).")
    opts = {**store.settings(), **(body.get("settings") or {})}
    chat = store.get_chat(body.get("chat_id") or "") or store.new_chat(E.model_id)
    msgs = chat["messages"]
    if body.get("truncate_at") is not None:
        del msgs[int(body["truncate_at"]):]
    text = str(body.get("message") or "").strip()
    for a in body.get("attachments") or []:
        text += f"\n\n<file name=\"{a.get('name', 'file')}\">\n{str(a.get('text', ''))[:60000]}\n</file>"
    if not text:
        raise RuntimeError("Type a message first.")
    msgs.append({"role": "user", "content": text, "ts": time.time()})
    if chat["title"] == "New chat":
        first = str(body.get("message") or text).strip().splitlines()[0]
        chat["title"] = (first[:52] + "…") if len(first) > 52 else first
    chat["model"] = E.model_id
    store.save_chat(chat)
    emit({"type": "meta", "chat_id": chat["id"], "title": chat["title"], "index": len(msgs) - 1})

    # Small models over-use tools, so only offer the tools this message plausibly needs.
    tools_ok = next((m.get("tools", True) for m in MODELS if m["id"] == E.model_id), True)
    schemas = (tools.pick(str(body.get("message") or ""), bool(opts.get("network_tools")))
               if opts.get("tools") and tools_ok else [])
    with_tools = bool(schemas)
    limit = min(int(opts.get("context_tokens", 4096)), E.max_context())
    reserve = int(opts.get("max_new_tokens", 768))
    history = [{"role": m["role"], "content": m["content"]} for m in msgs]
    convo, ids, trimmed = E.fit(_system_prompt(opts, with_tools), history, limit, reserve, schemas or None)
    emit({"type": "context", "used": int(ids.shape[1]), "limit": limit, "trimmed": trimmed})

    steps, stats, answer, seen = [], {}, "", set()
    for rnd in range(4):  # up to 3 tool calls; the last round never offers tools, so it must answer
        offer = with_tools and rnd < 3
        buf, mode = "", None

        def on_text(piece: str) -> None:
            nonlocal buf, mode
            buf += piece
            if mode is None:
                head = buf.lstrip()
                if offer and tools.looks_like_call(head) and len(head) < 12:
                    return
                mode = "tool" if offer and tools.looks_like_call(head) else "text"
                if mode == "text":
                    emit({"type": "token", "t": buf})
            elif mode == "text":
                emit({"type": "token", "t": piece})

        if rnd and not offer:  # final round: same conversation, tools removed
            ids = E.encode(convo)
        stats = E.stream(ids, opts, on_text)
        call = tools.parse_call(stats["text"]) if offer and mode != "text" else None
        key = call and json.dumps([call["name"], call["arguments"]], sort_keys=True)
        if call and key in seen:  # repeating itself: stop calling tools and answer
            with_tools = False
            continue
        if not call:
            answer = stats["text"].strip()
            if mode != "text" and answer:  # held back as a possible tool call, but it was plain text
                emit({"type": "token", "t": stats["text"]})
            if answer or stats.get("stopped") or not steps:
                break
            with_tools = False  # empty reply after tool use: ask once more without tools
            continue
        seen.add(key)
        emit({"type": "tool_call", "name": call["name"], "args": call["arguments"]})
        result = tools.run(call["name"], call["arguments"], bool(opts.get("network_tools")))
        steps.append({"name": call["name"], "args": call["arguments"], "result": result})
        emit({"type": "tool_result", "name": call["name"], "result": result})
        if isinstance(result, dict) and result.get("error"):
            with_tools = False  # a failed call: let the model answer with what it has instead of retrying
        convo = convo + [
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}]},
            {"role": "tool", "name": call["name"], "content": json.dumps(result, ensure_ascii=False)},
        ]
        ids = E.encode(convo, schemas)
        if stats.get("stopped"):
            break

    msgs.append({"role": "assistant", "content": answer, "ts": time.time(), "model": E.model_id,
                 "steps": steps, "stats": {k: stats.get(k) for k in ("tokens", "seconds", "tps", "stopped")}})
    store.save_chat(chat)
    emit({"type": "done", "stats": msgs[-1]["stats"], "index": len(msgs) - 1})


# --------------------------------------------------------------------------- #
# OpenAI-compatible API
# --------------------------------------------------------------------------- #
def _openai_messages(body: dict) -> List[dict]:
    out = []
    for m in body.get("messages") or []:
        c = m.get("content")
        if isinstance(c, list):  # [{type: text, text: ...}]
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        out.append({"role": m.get("role", "user"), "content": c or ""})
    return out


def openai_completion(body: dict, emit_chunk: Optional[Emit]) -> dict:
    want = body.get("model")
    if want and want != E.model_id and E.local_path(want):
        E.load(want)
    if E.model is None:
        raise RuntimeError("No model is loaded. Open ablate chat and load one, or pass a downloaded model id.")
    opts = {"temperature": body.get("temperature", 0.7), "top_p": body.get("top_p", 0.9),
            "max_new_tokens": body.get("max_tokens") or body.get("max_completion_tokens") or 768,
            "repetition_penalty": 1.1}
    ids = E.encode(_openai_messages(body))
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())

    def chunk(delta: dict, finish=None) -> dict:
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": E.model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    if emit_chunk:
        emit_chunk(chunk({"role": "assistant", "content": ""}))
    stats = E.stream(ids, opts, (lambda t: emit_chunk(chunk({"content": t}))) if emit_chunk else (lambda t: None))
    if emit_chunk:
        emit_chunk(chunk({}, "stop"))
    return {"id": cid, "object": "chat.completion", "created": created, "model": E.model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": stats["text"]},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": int(ids.shape[1]), "completion_tokens": stats["tokens"],
                      "total_tokens": int(ids.shape[1]) + stats["tokens"]}}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def models_state() -> List[dict]:
    out = []
    for m in MODELS + [{"id": r, "name": r.split("/")[-1], "custom": True} for r in E.downloads
                       if r not in [x["id"] for x in MODELS]]:
        out.append({**m, "downloaded": bool(E.local_path(m["id"])), "loaded": E.model_id == m["id"],
                    "download": E.download_status(m["id"])})
    return out


def _explain(e: Exception) -> str:
    msg = str(e) or type(e).__name__
    if "out of memory" in msg.lower() or "MPS backend out of memory" in msg:
        return "Not enough memory for this model. Close other apps or pick a smaller model."
    if "401" in msg or "403" in msg or "gated" in msg.lower():
        return "HuggingFace refused the download (the repo may be private or gated)."
    return msg[:400]


class Handler(BaseHTTPRequestHandler):
    server_version = "ablate-chat"
    share = False

    def log_message(self, fmt, *args):
        pass

    # -- helpers ------------------------------------------------------------ #
    def _local(self) -> bool:
        return self.client_address[0] in ("127.0.0.1", "::1", "::ffff:127.0.0.1")

    def _host_ok(self) -> bool:
        """Reject DNS-rebinding requests: the Host header must be this machine."""
        if self.share:
            return True
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return host in ("localhost", "127.0.0.1", "::1")

    def _send(self, code: int, raw: bytes, ctype: str, extra: Optional[dict] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _json(self, obj, code: int = 200, extra: Optional[dict] = None) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", extra)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def _session(self) -> Optional[str]:
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "ablate_chat" and v in SESSIONS:
                return v
        return None

    def _authed(self) -> bool:
        p = store.public_profile()
        if not p or not p["accepted_terms"]:
            return False
        if self._session():
            return True
        # loopback without a passcode is the single-user default: no sign-in needed
        return self._local() and not p["has_passcode"]

    def _api_authed(self) -> bool:
        if self._local():
            return True
        p = store.public_profile() or {}
        return (self.headers.get("Authorization") or "") == f"Bearer {p.get('api_key')}"

    def _static(self, name: str) -> None:
        types = {".html": "text/html; charset=utf-8", ".png": "image/png", ".svg": "image/svg+xml",
                 ".webp": "image/webp"}
        ext = os.path.splitext(name)[1]
        res = resources.files("ablate.chat").joinpath("static", name)
        if ext not in types or "/" in name or not res.is_file():
            return self._json({"error": "not found"}, 404)
        self._send(200, res.read_bytes(), types[ext],
                   {"Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                    "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'"}
                   if ext == ".html" else None)

    def _ndjson_start(self) -> Emit:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def emit(ev: dict) -> None:
            self.wfile.write((json.dumps(ev, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()
        return emit

    # -- GET ---------------------------------------------------------------- #
    def do_GET(self) -> None:
        if not self._host_ok():
            return self._json({"error": "bad host"}, 403)
        url = urllib.parse.urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                return self._static("index.html")
            if url.path in ("/terms", "/terms.html"):
                return self._static("terms.html")
            if url.path.startswith("/static/"):
                return self._static(url.path[len("/static/"):])
            if url.path == "/api/me":
                p = store.public_profile()
                return self._json({"profile": p and {k: p[k] for k in ("name", "has_passcode", "accepted_terms", "avatar")},
                                   "authed": self._authed(), "local": self._local()})
            if url.path == "/v1/models":
                if not self._api_authed():
                    return self._json({"error": {"message": "missing API key"}}, 401)
                return self._json({"object": "list", "data": [
                    {"id": m["id"], "object": "model", "owned_by": "ai-anytime"} for m in models_state()
                    if m["downloaded"]]})
            if not self._authed():
                return self._json({"error": "sign in first"}, 401)
            if url.path == "/api/state":
                return self._json({
                    "profile": store.public_profile(), "settings": store.settings(), "memories": store.memories(),
                    "models": models_state(), "loaded": E.model_id,
                    "device": E.device or E.pick()[0], "max_context": E.max_context() if E.model else None,
                    "chats": store.list_chats(), "busy": E.lock.locked(), "local": self._local(),
                    "tools_ok": next((m.get("tools", True) for m in MODELS if m["id"] == E.model_id), True),
                    "port": self.server.server_address[1]})
            if url.path == "/api/models":
                return self._json({"models": models_state(), "loaded": E.model_id})
            if url.path == "/api/search":
                q = (urllib.parse.parse_qs(url.query).get("q") or [""])[0]
                return self._json({"results": store.search_chats(q) if q.strip() else []})
            if url.path.startswith("/api/chats/"):
                c = store.get_chat(url.path.rsplit("/", 1)[-1], mark_opened=True)
                return self._json(c or {"error": "not found"}, 200 if c else 404)
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": _explain(e)}, 500)

    # -- POST --------------------------------------------------------------- #
    def do_POST(self) -> None:
        if not self._host_ok():
            return self._json({"error": "bad host"}, 403)
        path = urllib.parse.urlparse(self.path).path
        try:
            body = self._body()
        except Exception:
            return self._json({"error": "bad JSON"}, 400)
        try:
            if path.startswith("/v1/"):
                return self._openai(path, body)
            if path.startswith("/api/auth/"):
                return self._auth(path, body)
            if not self._authed():
                return self._json({"error": "sign in first"}, 401)
            host_only = path in ("/api/models/download", "/api/models/remove", "/api/models/load")
            if host_only and not self._local():
                return self._json({"error": "Only the computer running ablate chat can manage models."}, 403)

            if path == "/api/chat":
                emit = self._ndjson_start()
                if not E.lock.acquire(timeout=0.2):
                    emit({"type": "queued", "message": "Waiting for the model to finish another reply…"})
                    E.lock.acquire()
                try:
                    chat_turn(body, emit)
                except (BrokenPipeError, ConnectionResetError):
                    E.stop.set()
                except Exception as e:
                    traceback.print_exc()
                    try:
                        emit({"type": "error", "message": _explain(e)})
                    except Exception:
                        pass
                finally:
                    E.lock.release()
                return
            if path == "/api/stop":
                E.stop.set()
                return self._json({"ok": True})
            if path == "/api/models/download":
                E.download(str(body["id"]).strip())
                return self._json({"models": models_state()})
            if path == "/api/models/remove":
                E.remove(str(body["id"]))
                return self._json({"models": models_state(), "loaded": E.model_id})
            if path == "/api/models/load":
                with E.lock:
                    if body.get("id"):
                        E.load(str(body["id"]))
                    else:
                        E.unload()
                return self._json({"models": models_state(), "loaded": E.model_id,
                                   "device": E.device, "max_context": E.max_context() if E.model else None,
                                   "tools_ok": next((m.get("tools", True) for m in MODELS if m["id"] == E.model_id), True)})
            if path == "/api/settings":
                return self._json({"settings": store.save_settings(body)})
            if path == "/api/memories":
                return self._json({"memories": store.save_memories(body.get("memories") or [])})
            if path == "/api/chats/update":
                store.update_chat(str(body.get("id")), **{k: body[k] for k in ("title", "pinned") if k in body})
                return self._json({"chats": store.list_chats()})
            if path == "/api/chats/delete":
                store.delete_chat(str(body.get("id")))
                return self._json({"chats": store.list_chats()})
            if path == "/api/profile":
                avatar = body.get("avatar")
                if avatar and not (isinstance(avatar, str) and len(avatar) < 400_000 and
                                   avatar.startswith(("data:image/png;base64,", "data:image/jpeg;base64,",
                                                      "data:image/webp;base64,"))):
                    return self._json({"error": "Avatar must be a PNG, JPEG or WebP image under 300 KB."}, 400)
                store.update_profile(body.get("name"), body.get("passcode"), avatar)
                return self._json({"profile": store.public_profile()})
            return self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                self._json({"error": _explain(e)}, 500)
            except Exception:
                pass

    def _auth(self, path: str, body: dict) -> None:
        cookie = lambda tok: {"Set-Cookie": f"ablate_chat={tok}; HttpOnly; SameSite=Strict; Path=/"}
        p = store.profile()
        if path == "/api/auth/create":
            if p:
                return self._json({"error": "A profile already exists. Sign in instead."}, 409)
            if not self._local():
                return self._json({"error": "Create the profile on the computer running ablate chat."}, 403)
            store.create_profile(body.get("name", ""), body.get("passcode", ""))
            tok = secrets.token_urlsafe(24)
            SESSIONS[tok] = time.time()
            return self._json({"ok": True}, extra=cookie(tok))
        if path == "/api/auth/login":
            if not p:
                return self._json({"error": "No profile yet."}, 404)
            if not p.get("pass_hash") and not self._local():
                return self._json({"error": "Set a passcode in Settings before signing in from another device."}, 403)
            if not store.check_passcode(body.get("passcode", "")):
                time.sleep(0.6)
                return self._json({"error": "Wrong passcode."}, 401)
            tok = secrets.token_urlsafe(24)
            SESSIONS[tok] = time.time()
            return self._json({"ok": True}, extra=cookie(tok))
        if path == "/api/auth/terms":
            if not (self._session() or (self._local() and p and not p.get("pass_hash"))):
                return self._json({"error": "sign in first"}, 401)
            store.accept_terms()
            return self._json({"ok": True})
        if path == "/api/auth/logout":
            SESSIONS.pop(self._session() or "", None)
            return self._json({"ok": True}, extra={"Set-Cookie": "ablate_chat=; Max-Age=0; Path=/"})
        return self._json({"error": "not found"}, 404)

    def _openai(self, path: str, body: dict) -> None:
        if not self._api_authed():
            return self._json({"error": {"message": "missing or wrong API key"}}, 401)
        if path != "/v1/chat/completions":
            return self._json({"error": {"message": "not found"}}, 404)
        stream = bool(body.get("stream"))
        with E.lock:
            if not stream:
                try:
                    return self._json(openai_completion(body, None))
                except Exception as e:
                    return self._json({"error": {"message": _explain(e)}}, 400)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            def sse(obj: dict) -> None:
                self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()
            try:
                openai_completion(body, sse)
            except (BrokenPipeError, ConnectionResetError):
                E.stop.set()
                return
            except Exception as e:
                sse({"error": {"message": _explain(e)}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()


def serve(host: str = "127.0.0.1", port: int = 7861, model: Optional[str] = None,
          open_browser: bool = True) -> None:
    Handler.share = host not in ("127.0.0.1", "localhost", "::1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if model:
        if not E.local_path(model):
            print(f"Downloading {model} (one time)…")
            from huggingface_hub import snapshot_download
            snapshot_download(model, allow_patterns=ALLOW)
        print(f"Loading {model}…")
        E.load(model)
    srv = ThreadingHTTPServer((host, port), Handler)
    url = f"http://localhost:{port}"
    print(f"\n  ablate chat  ·  {url}")
    print(f"  data folder  ·  {store.HOME}")
    print("  100% local: prompts and chats never leave this computer.")
    if Handler.share:
        print("  --share is on: other devices on your network can sign in with your passcode.")
    print("  Press Ctrl+C to stop.\n")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
