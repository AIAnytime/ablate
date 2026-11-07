"""Model backends for ``ablate code``.

``LocalBackend`` runs a HuggingFace model in this process (the same engine as
``ablate chat``): download once, then fully offline. ``RemoteBackend`` talks to
any OpenAI-compatible server, typically a RunPod GPU pod running vLLM with a
bigger abliterated model. Connections are shared with ``ablate chat``.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from ..remote import OpenAIClient

OnText = Callable[[str], None]


class LocalBackend:
    small = True   # 1-3B on a laptop: edit mode by default

    def __init__(self, model: str, say: Callable[[str], None]) -> None:
        from ..chat.server import ALLOW, Engine
        self.name = model
        self.E = Engine()
        if not self.E.local_path(model):
            say(f"Downloading {model} once from HuggingFace (then it runs offline)…")
            from huggingface_hub import constants, snapshot_download
            from ..ui.server import _find_token
            constants.HF_HUB_DISABLE_XET = True
            snapshot_download(model, allow_patterns=ALLOW, token=_find_token())
        say(f"Loading {model}…")
        self.E.load(model)
        self.device = self.E.device

    def max_context(self) -> int:
        return self.E.max_context()

    def count(self, messages: List[dict], tools: Optional[List[dict]]) -> int:
        return int(self.E.encode(messages, tools or None).shape[1])

    def generate(self, messages: List[dict], tools: Optional[List[dict]], opts: dict, on_text: OnText
                 ) -> Tuple[str, Optional[List[dict]], dict]:
        ids = self.E.encode(messages, tools or None)
        stats = self.E.stream(ids, opts, on_text)
        stats["prompt_tokens"] = int(ids.shape[1])
        return stats["text"], None, stats

    def stop(self) -> None:
        self.E.stop.set()

    def close(self) -> None:
        self.E.unload()


class RemoteBackend(OpenAIClient):
    small = False  # assume a bigger self-hosted model: full agent mode by default

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None, context: Optional[int] = None,
                 name: Optional[str] = None) -> None:
        super().__init__(base_url, model, api_key, context, name)
        self.device = base_url.split("//", 1)[-1].split("/", 1)[0]

    def max_context(self) -> int:
        return self.context

    def close(self) -> None:
        pass


def saved_remotes() -> List[dict]:
    try:
        from ..chat import store
        return store.list_remotes(with_keys=True)
    except Exception:
        return []


def find_remote(ref: str) -> Optional[dict]:
    ref = (ref or "").strip().lower()
    for r in saved_remotes():
        if ref in (r["id"].lower(), r["name"].lower()):
            return r
    return None


def remote_backend(r: dict) -> RemoteBackend:
    return RemoteBackend(r["base_url"], r["model"], r.get("api_key"), r.get("context"),
                         name=f"{r['name']} · {r['model'].split('/')[-1]}")
