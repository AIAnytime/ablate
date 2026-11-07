"""Remote models: any OpenAI-compatible server, with RunPod discovery.

Used by both ``ablate chat`` and ``ablate code``. A typical setup is a RunPod
GPU pod running vLLM with one of the abliterated models:

    vllm serve ai-anytime/Llama-3.2-3B-Instruct-abliterated --port 8000 --api-key <secret> \\
        --enable-auto-tool-choice --tool-call-parser llama3_json

RunPod exposes the port at ``https://<pod-id>-8000.proxy.runpod.net``; that plus
``/v1`` is the base URL. Connections are stored in the local Ablate database
(see ``ablate.chat.store``) so one connection serves both apps.
"""
from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
import uuid
from typing import Callable, Dict, List, Optional, Tuple

OnText = Callable[[str], None]
TOOL_FALLBACK = (
    "You can call these tools. To call one, reply with only:\n"
    '<tool_call>{"name": "<tool name>", "arguments": {...}}</tool_call>\n'
    "Tools:\n")


def _ssl() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _http(url: str, data: Optional[dict] = None, headers: Optional[dict] = None, timeout: float = 20,
          method: Optional[str] = None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json", "User-Agent": "ablate/remote",
                                          **(headers or {})}, method=method)
    return urllib.request.urlopen(req, timeout=timeout, context=_ssl() if url.startswith("https") else None)


def normalize(base_url: str) -> str:
    """Accept 'host:8000', 'https://x/', 'https://x/v1/chat/completions' ... and return '.../v1'."""
    u = base_url.strip().rstrip("/")
    if not u.startswith(("http://", "https://")):
        u = ("http://" if u.startswith(("localhost", "127.", "192.168.", "10.")) else "https://") + u
    for tail in ("/chat/completions", "/completions", "/models"):
        if u.endswith(tail):
            u = u[: -len(tail)]
    return u if u.endswith("/v1") else u + "/v1"


def probe(base_url: str, api_key: Optional[str] = None) -> List[dict]:
    """GET /v1/models. Returns [{'id', 'context'}]; raises RuntimeError with a readable message."""
    url = normalize(base_url) + "/models"
    try:
        with _http(url, headers={"Authorization": f"Bearer {api_key or 'none'}"}) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError("The server refused the API key.")
        raise RuntimeError(f"The server answered HTTP {e.code} at {url}.")
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise RuntimeError(f"Could not reach {url} ({getattr(e, 'reason', e)}). Is the pod running and the port exposed?")
    return [{"id": m.get("id"), "context": m.get("max_model_len") or m.get("context_length")}
            for m in data.get("data", []) if m.get("id")]


# --------------------------------------------------------------------------- #
# RunPod
# --------------------------------------------------------------------------- #
def runpod_pods(api_key: str) -> List[dict]:
    """List the account's pods with their HTTP ports and proxy URLs."""
    pods: List[dict] = []
    try:
        with _http("https://rest.runpod.io/v1/pods", headers={"Authorization": f"Bearer {api_key}"}) as r:
            raw = json.loads(r.read().decode())
        for p in raw if isinstance(raw, list) else raw.get("pods", raw.get("data", [])):
            ports = []
            for spec in p.get("ports") or []:
                port, _, kind = str(spec).partition("/")
                if kind == "http" and port.isdigit():
                    ports.append(int(port))
            pods.append({"id": p.get("id"), "name": p.get("name") or p.get("id"),
                         "status": p.get("desiredStatus") or p.get("status") or "",
                         "image": p.get("image") or p.get("imageName") or "",
                         "gpu": ((p.get("machine") or {}).get("gpuDisplayName") or p.get("gpuTypeId") or ""),
                         "ports": ports})
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError("RunPod refused the API key. Create one at runpod.io → Settings → API Keys.")
        pods = _runpod_graphql(api_key)
    except (urllib.error.URLError, OSError, ValueError):
        pods = _runpod_graphql(api_key)
    for p in pods:
        p["urls"] = [f"https://{p['id']}-{port}.proxy.runpod.net/v1" for port in p["ports"]]
    return pods


def _runpod_graphql(api_key: str) -> List[dict]:
    q = ("query { myself { pods { id name desiredStatus imageName "
         "runtime { ports { privatePort type } } machine { gpuDisplayName } } } }")
    try:
        with _http("https://api.runpod.io/graphql", {"query": q}, headers={"Authorization": f"Bearer {api_key}"}) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError("RunPod refused the API key." if e.code in (401, 403) else f"RunPod answered HTTP {e.code}.")
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise RuntimeError(f"Could not reach RunPod ({getattr(e, 'reason', e)}).")
    if data.get("errors"):
        raise RuntimeError("RunPod: " + str(data["errors"][0].get("message", data["errors"][0])))
    out = []
    for p in ((data.get("data") or {}).get("myself") or {}).get("pods") or []:
        ports = sorted({pt["privatePort"] for pt in ((p.get("runtime") or {}).get("ports") or [])
                        if pt.get("type") == "http"})
        out.append({"id": p["id"], "name": p.get("name") or p["id"], "status": p.get("desiredStatus", ""),
                    "image": p.get("imageName", ""), "gpu": (p.get("machine") or {}).get("gpuDisplayName", ""),
                    "ports": ports})
    return out


# --------------------------------------------------------------------------- #
# Chat client (streaming, native tool calls, text fallback)
# --------------------------------------------------------------------------- #
class OpenAIClient:
    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None, context: Optional[int] = None,
                 name: Optional[str] = None) -> None:
        self.base = normalize(base_url)
        self.model = model
        self.key = api_key or "none"
        self.context = int(context or 32768)
        self.name = name or model
        self.native_tools = True   # flips off if the server rejects the tools parameter
        self._stop = False

    def count(self, messages: List[dict], tools: Optional[List[dict]] = None) -> int:
        return int(len(json.dumps(messages, ensure_ascii=False)) / 3.6 +
                   (len(json.dumps(tools)) / 3.6 if tools else 0)) + 16

    @staticmethod
    def wire(messages: List[dict]) -> List[dict]:
        """Internal messages -> OpenAI wire format (tool call ids, JSON-string arguments)."""
        out, pending = [], []
        for m in messages:
            if m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    cid = c.get("id") or "call_" + uuid.uuid4().hex[:12]
                    c["id"] = cid
                    pending.append(cid)
                    args = c["function"]["arguments"]
                    calls.append({"id": cid, "type": "function", "function": {
                        "name": c["function"]["name"], "arguments": args if isinstance(args, str) else json.dumps(args)}})
                out.append({"role": "assistant", "content": m.get("content") or None, "tool_calls": calls})
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": pending.pop(0) if pending else "call_0",
                            "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m.get("content") or ""})
        return out

    @staticmethod
    def text_wire(messages: List[dict], tools: Optional[List[dict]]) -> List[dict]:
        """For servers without tool support: describe tools in the system prompt and flatten tool turns."""
        out = []
        for m in messages:
            if m["role"] == "system" and tools:
                desc = "\n".join(f"- {t['function']['name']}: {t['function']['description']} "
                                 f"args: {', '.join(t['function']['parameters'].get('properties', {}))}" for t in tools)
                out.append({"role": "system", "content": m["content"] + "\n\n" + TOOL_FALLBACK + desc})
            elif m.get("tool_calls"):
                c = m["tool_calls"][0]["function"]
                out.append({"role": "assistant", "content": (m.get("content") or "") + "\n<tool_call>" +
                            json.dumps({"name": c["name"], "arguments": c["arguments"]}) + "</tool_call>"})
            elif m["role"] == "tool":
                out.append({"role": "user", "content": f"Result of {m.get('name', 'the tool')}:\n{m['content']}"})
            else:
                out.append({"role": m["role"], "content": m.get("content") or ""})
        return out

    def generate(self, messages: List[dict], tools: Optional[List[dict]], opts: dict, on_text: OnText
                 ) -> Tuple[str, Optional[List[dict]], dict]:
        """Stream a reply. Returns (text, native tool calls or None, stats)."""
        self._stop = False
        use_native = bool(tools) and self.native_tools
        body = {"model": self.model, "stream": True,
                "messages": self.wire(messages) if (use_native or not tools) else self.text_wire(messages, tools),
                "temperature": float(opts.get("temperature", 0.7)), "top_p": float(opts.get("top_p", 0.95)),
                "max_tokens": int(opts.get("max_new_tokens", 768))}
        if use_native:
            body["tools"] = tools
        try:
            r = _http(self.base + "/chat/completions", body, headers={"Authorization": f"Bearer {self.key}"},
                      timeout=600)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            if use_native and e.code in (400, 422) and "tool" in detail.lower():
                self.native_tools = False   # e.g. vLLM without --enable-auto-tool-choice: retry with text tools
                return self.generate(messages, tools, opts, on_text)
            if e.code in (401, 403):
                raise RuntimeError("The remote server refused the API key.")
            raise RuntimeError(f"Remote server error {e.code}: {detail}")
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"Could not reach {self.base} ({getattr(e, 'reason', e)}). Is the pod still running?")
        text, calls, finish = "", {}, None
        with r:
            for raw in r:
                if self._stop:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    choice = json.loads(data)["choices"][0]
                except (ValueError, KeyError, IndexError):
                    continue
                delta = choice.get("delta") or {}
                finish = choice.get("finish_reason") or finish
                if delta.get("content"):
                    text += delta["content"]
                    on_text(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"name": "", "args": ""})
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["args"] += fn.get("arguments") or ""
        native = []
        for slot in calls.values():
            try:
                args = json.loads(slot["args"] or "{}")
            except ValueError:
                args = {}
            native.append({"name": slot["name"], "arguments": args})
        return text, (native or None), {"text": text, "tokens": max(1, int(len(text) / 3.6)) if text else 0,
                                         "stopped": self._stop, "finish": finish}

    def stop(self) -> None:
        self._stop = True
