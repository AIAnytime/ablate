"""Tools the chat models can call, plus a parser for their tool-call output.

Every tool runs on this machine. ``fetch_url`` is the only one that touches the
network, and it is off unless the user turns on "network tools" in settings.
"""
from __future__ import annotations

import ast
import datetime as _dt
import html
import ipaddress
import json
import math
import operator
import re
import socket
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

from . import store

# -- calculator ------------------------------------------------------------ #
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {name: getattr(math, name) for name in
          ("sqrt", "log", "log10", "log2", "exp", "sin", "cos", "tan", "floor", "ceil", "factorial")}
_FUNCS.update(abs=abs, round=round, min=min, max=max)
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("only arithmetic and math functions are allowed")


def calculator(expression: str) -> dict:
    expr = str(expression).replace("^", "**").replace("×", "*").replace("÷", "/").replace(",", "")
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"\1/100*", expr)
    return {"expression": expression, "result": _eval(ast.parse(expr, mode="eval"))}


def current_time(timezone: str = "") -> dict:
    now = _dt.datetime.now().astimezone()
    if timezone:
        try:
            from zoneinfo import ZoneInfo
            now = _dt.datetime.now(ZoneInfo(timezone))
        except Exception:
            pass
    return {"iso": now.isoformat(timespec="seconds"), "weekday": now.strftime("%A"),
            "timezone": str(now.tzinfo)}


def search_chats(query: str) -> dict:
    return {"query": query, "results": store.search_chats(query)}


def _public_host(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname or ""
    try:
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        return True
    except (OSError, ValueError):
        return False


def fetch_url(url: str) -> dict:
    if not re.match(r"^https?://", str(url)):
        url = "https://" + str(url)
    if not _public_host(url):
        return {"error": "only public web addresses can be fetched"}
    from ..ui.server import _ssl_context
    req = urllib.request.Request(url, headers={"User-Agent": "ablate-chat/1.0"})
    with urllib.request.urlopen(req, timeout=12, context=_ssl_context()) as r:
        raw = r.read(600_000).decode(r.headers.get_content_charset() or "utf-8", "replace")
    raw = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", raw)
    title = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"(?s)<[^>]+>", " ", raw))).strip()
    return {"url": url, "title": html.unescape(title.group(1).strip()) if title else "", "text": text[:4000]}


TOOLS: Dict[str, Tuple[Callable, dict, bool]] = {
    # name: (function, JSON schema, needs network)
    "calculator": (calculator, {
        "type": "function", "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression exactly, e.g. '18/100*2450' or 'sqrt(2)*10'.",
            "parameters": {"type": "object", "properties": {
                "expression": {"type": "string", "description": "The arithmetic expression."}},
                "required": ["expression"]}}}, False),
    "current_time": (current_time, {
        "type": "function", "function": {
            "name": "current_time",
            "description": "Get the current local date and time, optionally in an IANA timezone like 'Asia/Dubai'.",
            "parameters": {"type": "object", "properties": {
                "timezone": {"type": "string", "description": "Optional IANA timezone name."}}}}}, False),
    "search_chats": (search_chats, {
        "type": "function", "function": {
            "name": "search_chats",
            "description": "Search the user's earlier conversations saved on this computer.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Words to look for."}},
                "required": ["query"]}}}, False),
    "fetch_url": (fetch_url, {
        "type": "function", "function": {
            "name": "fetch_url",
            "description": "Download a public web page and return its title and text.",
            "parameters": {"type": "object", "properties": {
                "url": {"type": "string", "description": "The page address."}},
                "required": ["url"]}}}, True),
}


def schemas(network: bool) -> List[dict]:
    return [schema for _, schema, net in TOOLS.values() if network or not net]


def run(name: str, args: dict, network: bool) -> dict:
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    fn, _, net = TOOLS[name]
    if net and not network:
        return {"error": "network tools are turned off in settings"}
    try:
        return fn(**{k: v for k, v in (args or {}).items() if isinstance(k, str)})
    except Exception as e:  # tools report errors back to the model instead of crashing the chat
        return {"error": f"{type(e).__name__}: {e}"}


def parse_call(text: str) -> Optional[dict]:
    """Return ``{"name", "arguments"}`` if ``text`` is a tool call, else None.

    Handles Qwen's ``<tool_call>{...}</tool_call>`` and Llama 3's bare JSON
    ``{"name": ..., "parameters": {...}}`` (optionally after ``<|python_tag|>``).
    """
    t = text.strip().replace("<|python_tag|>", "").strip()
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", t, re.S)
    blob = m.group(1) if m else (t if t.startswith("{") else None)
    if blob is None:
        return None
    try:
        obj = json.loads(blob)
    except ValueError:
        try:
            obj = json.loads(blob[: blob.rfind("}") + 1])
        except ValueError:
            return None
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function")
    name = obj.get("name") or (fn.get("name") if isinstance(fn, dict) else fn)
    args = obj.get("arguments", obj.get("parameters"))
    if args is None and isinstance(fn, dict):
        args = fn.get("arguments", fn.get("parameters"))
    args = args if args is not None else {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    if name not in TOOLS:
        return None
    return {"name": name, "arguments": args if isinstance(args, dict) else {}}


def looks_like_call(prefix: str) -> bool:
    """True when the first characters of a reply suggest a tool call is coming."""
    p = prefix.lstrip().replace("<|python_tag|>", "")
    return p.startswith("{") or p.startswith("<tool_call>") or "<tool_call>".startswith(p[:11]) and p.startswith("<")
