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
names = lambda m: [x["function"]["name"] for x in tools.pick(m, False)]
check("routing: creative prompt gets no tools", names("Write a chilling monologue for my villain") == [])
check("routing: maths gets the calculator", names("What is 18% of 2,450?") == ["calculator"])
check("routing: date gets the clock", names("what's the date today?") == ["current_time"])
check("routing: web fetch stays off without network", names("summarise https://example.com") == [])
store.update_profile(avatar="data:image/png;base64,AAAA")
check("avatar saved", store.public_profile()["avatar"].startswith("data:image/png"))
store.save_settings({"theme": "light"})
check("theme persisted", store.settings()["theme"] == "light")

# one-time import from the 0.4.0 JSON layout
import importlib, json as _json, pathlib
old = tempfile.mkdtemp(prefix="ablate-chat-legacy-")
home = pathlib.Path(old) / "chat"; (home / "chats").mkdir(parents=True)
(home / "profile.json").write_text(_json.dumps({"name": "Legacy", "salt": "00" * 16, "pass_hash": "", "accepted_terms_at": 1.0, "api_key": "k"}))
(home / "chats" / "abc.json").write_text(_json.dumps({"id": "abc", "title": "Old chat", "created": 1, "updated": 2,
    "messages": [{"role": "user", "content": "hello from json"}, {"role": "assistant", "content": "hi", "stats": {"tokens": 1}}]}))
os.environ["ABLATE_HOME"] = old
legacy = importlib.reload(store)
check("legacy profile imported", legacy.public_profile()["name"] == "Legacy")
check("legacy chat imported", legacy.get_chat("abc")["messages"][1]["stats"]["tokens"] == 1)
check("legacy chat searchable", legacy.search_chats("from json")[0]["id"] == "abc")
print("\nALL CHAT CHECKS PASSED")
