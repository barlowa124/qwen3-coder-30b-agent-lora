#!/usr/bin/env python3
"""Replay the actual failing turns against :8015/ollama/v1/chat/completions.

Simulates the OpenCode tool loop: send user + tools -> if tool_calls,
execute whitelisted commands for real (read-only), feed tool result back,
repeat up to N steps; score per case.
"""
import json, subprocess, sys, urllib.request, shlex, os, time

URL = "http://127.0.0.1:8015/ollama/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:14b"
GMAIL_PY = "/home/asdf/.local/share/gmail-mcp/venv/bin/python"
GMAIL_CMD = (GMAIL_PY + ' -c "import sys, json; '
             'sys.path.insert(0, \'/home/asdf/.local/share/gmail-mcp\'); '
             'import server; print(json.dumps(server.gmail_list_recent(), indent=2))"')

TOOLS = [
    {"type": "function", "function": {
        "name": "shell",
        "description": "Run a shell command and return stdout+stderr.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read",
        "description": "Read a file from the filesystem.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write",
        "description": "Write a file to the filesystem.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"}},
                       "required": ["pattern"]}}},
]

SAFE_PREFIX = (
    "ls", "find", "cat", "head", "tail", "echo PWD", "pwd",
    "curl -s http://[::1]:9222", "curl -s http://127.0.0.1:922", "curl -s http://localhost:922",
    "bluetoothctl", "coredumpctl", "journalctl", "systemctl status bluetooth", "systemctl --user status bluetooth",
    "/home/asdf/.local/bin/devin-quota",
    GMAIL_PY, "/home/asdf/.local/share/gmail-mcp/venv/bin/python", "cd /home/asdf/.local/share/gmail-mcp",
)

def run_tool(name, args):
    if name == "read":
        p = args.get("path", "")
        try:
            return open(os.path.expanduser(p)).read()[:4000]
        except Exception as e:
            return str(e)
    if name == "glob":
        import glob as g
        r = g.glob(os.path.expanduser(args.get("pattern", "")))[:50]
        return "\n".join(r) or "no matches"
    if name == "write":
        return "ok (sandboxed: not written)"
    if name == "shell":
        cmd = args.get("command", "")
        if not any(cmd.strip().startswith(p) for p in SAFE_PREFIX):
            return "ok"
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True,
                               text=True, timeout=30)
            return (r.stdout + r.stderr)[:20000]
        except Exception as e:
            return str(e)
    return "unknown tool"

def call(msgs):
    body = {"model": MODEL, "messages": msgs, "tools": TOOLS,
            "stream": False}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())

PRESET = {
    # after the "modify opencode" turn, inject a failing tool episode like the
    # real turn-1556 require('fs') error before the follow-up user message
    "fix_errors:1": [
        {"role": "assistant", "tool_calls": [
            {"id": "x1", "type": "function",
             "function": {"name": "read",
                          "arguments": '{"path": "/home/asdf/.config/opencode/config.json"}'}}]},
        {"role": "tool", "tool_call_id": "x1",
         "content": "File not found: /home/asdf/.config/opencode/config.json"},
        {"role": "assistant", "tool_calls": [
            {"id": "x2", "type": "function",
             "function": {"name": "write",
                          "arguments": '{"path": "/home/asdf/.config/opencode/config.json", "content": "{ // comment\\n}"}'}}]},
        {"role": "tool", "tool_call_id": "x2",
         "content": "Error: JSON validation failed: unexpected character at line 1"},
    ],
}

CASES = [
    # (name, turns) -- each turn a user message; we run a mini agent loop
    ("emails", ["check my emails and summarize them"]),
    ("firefox", ["open firefox"]),
    ("fs", ["do you see my file system"]),
    ("chromium_cdp", ["delete the chats in the chatgpt webui",
                      "you have chromium"]),
    ("fix_errors", ["modify opencode so it unloads qwen when i swap models",
                    "fix the erros that happened"]),
    ("continuation", ["the bluetooth isnt connecting to the headphones the bluetooth process is crashing",
                      "conitnue"]),
    ("quota", ["i need to see my devin quota to tell when the creidts are about to be exhausted or im on over limit credits in opencode"]),
    ("multistep", ["check my emails and read the most recent one fully"]),
]

FILTER = sys.argv[2] if len(sys.argv) > 2 else None
for name, turns in CASES:
    if FILTER and FILTER not in name:
        continue
    msgs = [{"role": "system",
             "content": open("/home/asdf/.config/opencode/agents/local.md")
                        .read()}]
    steps = 0
    calls = []
    for i, t in enumerate(turns):
        msgs.append({"role": "user", "content": t})
        preset = PRESET.get(f"{name}:{i}")
        if preset:
            msgs.extend(preset)
        for _ in range(6):
            r = call(msgs)
            m = r["choices"][0]["message"]
            msgs.append(m)
            steps += 1
            tc = m.get("tool_calls") or []
            if not tc:
                break
            for c in tc:
                fn = c["function"]["name"]
                try:
                    ar = json.loads(c["function"].get("arguments") or "{}")
                except Exception:
                    ar = {}
                calls.append((fn, ar))
                msgs.append({"role": "tool",
                             "tool_call_id": c.get("id", "x"),
                             "content": run_tool(fn, ar)})
    final = msgs[-1].get("content") or ""
    print(f"\n===== {name} =====")
    print("calls:", [(n, str(a)[:90]) for n, a in calls])
    print("final:", final[:300].replace("\n", " "))
