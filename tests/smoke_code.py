"""Model-free checks for ``ablate code``: parsing, tools, permissions, skills, MCP, context budget.

    python tests/smoke_code.py
"""
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

os.environ["ABLATE_HOME"] = tempfile.mkdtemp(prefix="ablate-code-test-")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ablate.code import agent as A  # noqa: E402
from ablate.code.tools import SCHEMAS  # noqa: E402,F401
from ablate.code import config as C  # noqa: E402
from ablate.code import skills as SK  # noqa: E402
from ablate.code.mcp import MCPHub  # noqa: E402
from ablate.code.tools import Workspace, command_allowed  # noqa: E402
from ablate.code.ui import UI  # noqa: E402


def check(name, cond):
    print(("[ok] " if cond else "[FAIL] ") + name)
    if not cond:
        sys.exit(1)


names = {"read_file", "edit_file", "bash", "grep"}
calls, prose = A._calls('I will look.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>', names)
check("parse Qwen <tool_call>", calls == [{"name": "read_file", "arguments": {"path": "a.py"}}] and prose == "I will look.")
check("parse Llama bare JSON", A._calls('{"name": "bash", "parameters": {"command": "ls"}}', names)[0][0]["arguments"] == {"command": "ls"})
check("parse fenced json", A._calls('```json\n{"name": "grep", "arguments": {"pattern": "x"}}\n```', names)[0][0]["name"] == "grep")
check("ignore unknown tool", A._calls('{"name": "rm_everything", "arguments": {}}', names)[0] == [])
check("plain answer has no calls", A._calls("All done: fixed the bug.", names) == ([], "All done: fixed the bug."))

shown = []
f = A.StreamFilter(shown.append)
for piece in ["Looking ", "at it.\n<tool", "_call>{\"name\": \"bash\"}</tool_call>"]:
    f.feed(piece)
f.flush()
check("stream filter hides tool JSON", "".join(shown) == "Looking at it.\n")
shown.clear()
g = A.StreamFilter(shown.append)
g.feed('{"name": "bash", "arguments": {}}')
g.flush()
check("stream filter hides bare JSON call", shown == [])

root = Path(tempfile.mkdtemp(prefix="ablate-code-proj-"))
(root / "src").mkdir()
(root / "src" / "calc.py").write_text(textwrap.dedent("""\
    def add(a, b):
        return a - b


    def mul(a, b):
        return a * b
    """))
(root / "tests").mkdir()
(root / "tests" / "test_calc.py").write_text("from src.calc import add\n\ndef test_add():\n    assert add(2, 2) == 4\n")
ws = Workspace(root)
check("files listed", "src/calc.py" in ws.files())
check("repo map", "src/" in ws.repo_map())
check("tests detected", ws.detect_tests() == "python -m pytest -q")
check("grep", "src/calc.py:2:" in ws.grep("return a - b"))
check("read_file numbers lines", "    2      return a - b" in ws.read_file("src/calc.py"))
check("read_file survives nonsense range", "def add" in ws.read_file("src/calc.py", 0, -1))
check("read_file suggests close paths", "Did you mean" in ws.read_file("src/calk.py"))
msg, change = ws.edit_file("src/calc.py", "return a - b", "return a + b")
check("exact edit", change and "a + b" in (root / "src" / "calc.py").read_text())
msg, change = ws.edit_file("src/calc.py", "def mul(a,b):\nreturn a*b", "x")
check("bad edit explains and shows closest lines", change is None and "Closest lines" in msg)
msg, change = ws.edit_file("src/calc.py", "  return a * b", "    return a * b * 1")
check("whitespace-insensitive edit keeps indentation", change and "    return a * b * 1" in (root / "src" / "calc.py").read_text())
check("undo restores", "Restored" in ws.revert() and "return a * b\n" in (root / "src" / "calc.py").read_text())
try:
    ws.path("../../etc/passwd")
    check("paths outside the project are refused", False)
except ValueError:
    check("paths outside the project are refused", True)
check("bash runs in project", ws.bash("ls src").startswith("exit code 0") and "calc.py" in ws.bash("ls src"))
check("relevant finds files by name", "src/calc.py" in ws.relevant("fix the add function in calc"))
check("preload gives numbered source", "Contents of src/calc.py" in ws.preload(ws.relevant("fix calc")))
ws.note("plan: fix add")
check("scratchpad persists", "plan: fix add" in ws.scratchpad())

allow = C.DEFAULTS["allow"]
check("allowlisted command", command_allowed("git status", allow, []) is True)
check("unknown command asks", command_allowed("python script.py", allow, []) is None)
check("chained command asks", command_allowed("ls && rm x", allow, []) is None)
check("destructive command never auto", command_allowed("rm -rf /", allow + ["rm*"], []) is None)
check("deny list wins", command_allowed("git status", allow, ["git*"]) is False)

sk = root / ".ablate" / "skills" / "release"
sk.mkdir(parents=True)
(sk / "SKILL.md").write_text("---\nname: release\ndescription: How to cut a release\n---\nBump the version, then tag.\n")
found = SK.find(root)
check("skill found", "release" in found and "cut a release" in SK.listing(found))
check("skill body loads without front matter", SK.load(found, "release").startswith("Bump the version"))

server = root / "echo_mcp.py"
server.write_text(textwrap.dedent('''\
    import json, sys
    for line in sys.stdin:
        m = json.loads(line)
        if "id" not in m:
            continue
        if m["method"] == "initialize":
            r = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "echo", "version": "1"}}
        elif m["method"] == "tools/list":
            r = {"tools": [{"name": "echo", "description": "Echo text back", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]}
        elif m["method"] == "tools/call":
            r = {"content": [{"type": "text", "text": "echo: " + m["params"]["arguments"].get("text", "")}]}
        else:
            r = {}
        print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}), flush=True)
    '''))
hub = MCPHub({"echo": {"command": sys.executable, "args": [str(server)]}})
check("mcp lists tools", "- echo(text): Echo text back" in hub.run("echo"))
check("mcp calls a tool", hub.run("echo", "echo", {"text": "hi"}) == "echo: hi")
check("mcp unknown server", "Unknown MCP server" in hub.run("nope"))
hub.close()


class FakeBackend:
    name, device = "fake", "cpu"

    def max_context(self):
        return 100_000

    def count(self, messages, tools):
        return sum(len(m.get("content") or "") for m in messages) // 4 + 200

    def generate(self, messages, tools, opts, on_text):
        return "", None, {}

    def stop(self):
        pass


cfg = C.load(root, {"context": 2000, "max_new_tokens": 300})
ag = A.Agent(root, cfg, UI("mono", color=False), FakeBackend())
ag.history = [{"role": "user", "content": "fix the add bug"}]
for i in range(12):
    ag.history.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": {"path": f"f{i}.py"}}}]})
    ag.history.append({"role": "tool", "name": "read_file", "content": f"result {i} " + "x" * 1500})
msgs, used = ag.build(0)
check("context fits the budget", used + 300 <= 2000)
check("compaction keeps the goal", msgs[1]["content"] == "fix the add bug")
check("latest result kept", any(f"result 11" in (m.get("content") or "") for m in msgs))
# -- edit mode parsing -------------------------------------------------------------- #
from ablate.code.editor import parse_blocks, outside_blocks  # noqa: E402
reply = ("app/config.py\n<<<<<<< SEARCH\n    header = lines[0]\n=======\n    if not lines:\n        return settings\n"
         "    header = lines[0]\n>>>>>>> REPLACE\nHandle empty files.")
blocks = parse_blocks(reply, None)
check("edit block parsed", blocks == [("app/config.py", "    header = lines[0]", "    if not lines:\n        return settings\n    header = lines[0]")])
check("text outside blocks kept", outside_blocks(reply) == "Handle empty files.")
check("missing path falls back to the shown file", parse_blocks("<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE", "a.py")[0][0] == "a.py")
check("fenced and short markers tolerated", len(parse_blocks("`b.py`\n```python\n<<<<< SEARCH\nx\n=====\ny\n>>>>> REPLACE\n```", None)) == 1)

# -- remote: URL normalising and RunPod discovery (mocked API) ---------------------------- #
import io  # noqa: E402
from ablate import remote as R  # noqa: E402
check("normalize adds /v1", R.normalize("abc-8000.proxy.runpod.net") == "https://abc-8000.proxy.runpod.net/v1")
check("normalize strips endpoint", R.normalize("http://localhost:8000/v1/chat/completions") == "http://localhost:8000/v1")


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


calls = []


def fake_http(url, data=None, headers=None, timeout=20, method=None):
    calls.append(url)
    if "rest.runpod.io" in url:
        return _Resp(json.dumps([{"id": "pod123", "name": "vllm-3b", "desiredStatus": "RUNNING", "image": "vllm/vllm-openai",
                                  "ports": ["8000/http", "22/tcp"], "machine": {"gpuDisplayName": "A40"}}]).encode())
    raise AssertionError(url)


R._http = fake_http
pods = R.runpod_pods("rp_test")
check("runpod pods parsed", pods[0]["id"] == "pod123" and pods[0]["ports"] == [8000] and pods[0]["gpu"] == "A40")
check("runpod proxy url built", pods[0]["urls"] == ["https://pod123-8000.proxy.runpod.net/v1"])


def fake_http_gql(url, data=None, headers=None, timeout=20, method=None):
    if "rest.runpod.io" in url:
        raise OSError("rest unavailable")
    return _Resp(json.dumps({"data": {"myself": {"pods": [{"id": "p9", "name": "old", "desiredStatus": "EXITED",
            "imageName": "img", "runtime": {"ports": [{"privatePort": 8000, "type": "http"}]}, "machine": None}]}}}).encode())


R._http = fake_http_gql
check("graphql fallback", R.runpod_pods("k")[0]["urls"] == ["https://p9-8000.proxy.runpod.net/v1"])
text_msgs = R.OpenAIClient.text_wire([{"role": "system", "content": "sys"},
                                       {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "grep", "arguments": {"pattern": "x"}}}]},
                                       {"role": "tool", "name": "grep", "content": "a.py:1: x"}], [SCHEMAS["grep"]])
check("text tool fallback describes tools", "<tool_call>" in text_msgs[0]["content"] and "grep" in text_msgs[0]["content"])
check("text tool fallback flattens results", text_msgs[2] == {"role": "user", "content": "Result of grep:\na.py:1: x"})
print("\nALL CODE CHECKS PASSED")
