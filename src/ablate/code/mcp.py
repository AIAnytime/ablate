"""A minimal MCP (Model Context Protocol) client over stdio.

Servers are configured in config.json under ``mcp.servers``:

    {"mcp": {"servers": {"fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}}}}

A server starts only the first time the model asks for it, and its tool list is
fetched on demand. Small models cannot afford every MCP tool schema in every
prompt, so they are reached through one ``mcp`` tool instead.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from typing import Any, Dict, List, Optional

PROTOCOL = "2025-06-18"


class MCPError(RuntimeError):
    pass


class MCPServer:
    def __init__(self, name: str, spec: dict) -> None:
        self.name, self.spec = name, spec
        self.proc: Optional[subprocess.Popen] = None
        self._id = 0
        self._replies: "queue.Queue[dict]" = queue.Queue()
        self.tools: Optional[List[dict]] = None

    def start(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        cmd = [self.spec["command"], *self.spec.get("args", [])]
        env = {**os.environ, **{k: str(v) for k, v in (self.spec.get("env") or {}).items()}}
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1, env=env, cwd=self.spec.get("cwd"))
        threading.Thread(target=self._read, daemon=True).start()
        self._request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                     "clientInfo": {"name": "ablate-code", "version": "0.5"}}, timeout=30)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _read(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                self._replies.put(msg)

    def _send(self, msg: dict) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _request(self, method: str, params: dict, timeout: float = 60) -> Any:
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        while True:
            try:
                msg = self._replies.get(timeout=timeout)
            except queue.Empty:
                raise MCPError(f"{self.name}: no reply to {method} within {timeout:.0f}s")
            if msg.get("id") != rid:
                continue
            if "error" in msg:
                raise MCPError(f"{self.name}: {msg['error'].get('message', msg['error'])}")
            return msg["result"]

    def list_tools(self) -> List[dict]:
        self.start()
        if self.tools is None:
            self.tools = self._request("tools/list", {}).get("tools", [])
        return self.tools

    def call(self, tool: str, arguments: dict) -> str:
        self.start()
        res = self._request("tools/call", {"name": tool, "arguments": arguments or {}}, timeout=120)
        parts = []
        for c in res.get("content", []):
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
            else:
                parts.append(f"[{c.get('type')} content]")
        text = "\n".join(parts) or json.dumps(res.get("structuredContent", {}))[:4000]
        return ("Error from tool: " if res.get("isError") else "") + text

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


class MCPHub:
    def __init__(self, servers: Dict[str, dict]) -> None:
        self.servers = {n: MCPServer(n, s) for n, s in (servers or {}).items() if s.get("command")}

    def __bool__(self) -> bool:
        return bool(self.servers)

    def names(self) -> List[str]:
        return list(self.servers)

    def run(self, server: str, tool: Optional[str] = None, arguments: Optional[dict] = None) -> str:
        if server not in self.servers:
            return f"Unknown MCP server {server!r}. Configured: {', '.join(self.servers) or 'none'}."
        s = self.servers[server]
        try:
            if not tool:
                tools = s.list_tools()
                rows = []
                for t in tools:
                    props = ", ".join((t.get("inputSchema") or {}).get("properties", {}).keys())
                    rows.append(f"- {t['name']}({props}): {(t.get('description') or '').strip().splitlines()[0][:120] if t.get('description') else ''}")
                return f"Tools on {server}:\n" + "\n".join(rows)
            return s.call(tool, arguments or {})
        except (MCPError, OSError, ValueError) as e:
            return f"MCP error: {e}"

    def close(self) -> None:
        for s in self.servers.values():
            s.close()
