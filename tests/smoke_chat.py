"""Model-free checks for ``ablate chat``: tools, tool-call parsing, local storage.

    python tests/smoke_chat.py
"""
import os
import sys
import tempfile

os.environ["ABLATE_HOME"] = tempfile.mkdtemp(prefix="ablate-chat-test-")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ablate.chat import store, tools  # noqa: E402


def check(name, cond):
    print(("[ok] " if cond else "[FAIL] ") + name)
    if not cond:
        sys.exit(1)


check("calculator: percent of", tools.calculator("18% of 2,450")["result"] == 441.0)
check("calculator: functions", round(tools.calculator("sqrt(16)+2^3")["result"]) == 12)
check("calculator rejects code", "error" in tools.run("calculator", {"expression": "__import__('os')"}, False))
check("network tool blocked when off", "error" in tools.run("fetch_url", {"url": "https://example.com"}, False))
check("fetch_url refuses private hosts", "error" in tools.fetch_url("http://127.0.0.1:7861/"))
check("schemas hide network tools", all(s["function"]["name"] != "fetch_url" for s in tools.schemas(False)))

check("parse Qwen <tool_call>", tools.parse_call('<tool_call>\n{"name": "calculator", "arguments": {"expression": "1+1"}}\n</tool_call>')
      == {"name": "calculator", "arguments": {"expression": "1+1"}})
check("parse Llama JSON", tools.parse_call('{"name": "current_time", "parameters": {}}')["name"] == "current_time")
check("parse function-as-string", tools.parse_call('{"type": "function", "function": "calculator", "parameters": {"expression": "2"}}')["name"] == "calculator")
check("plain JSON is not a call", tools.parse_call('{"answer": 42}') is None)
check("plain text is not a call", tools.parse_call("Hello there") is None)

p = store.create_profile("Tester", "secret")
check("passcode verifies", store.check_passcode("secret") and not store.check_passcode("nope"))
check("api key issued", store.public_profile()["api_key"].startswith("abl-local-"))
c = store.new_chat("m")
c["messages"].append({"role": "user", "content": "the zebra crossing"})
store.save_chat(c)
check("chat saved and listed", store.list_chats()[0]["id"] == c["id"])
check("search finds chat", store.search_chats("zebra")[0]["role"] == "user")
check("settings clamp to known keys", "bogus" not in store.save_settings({"temperature": 0.2, "bogus": 1}))
store.delete_chat(c["id"])
check("chat deleted", not store.list_chats())
print("\nALL CHAT CHECKS PASSED")
