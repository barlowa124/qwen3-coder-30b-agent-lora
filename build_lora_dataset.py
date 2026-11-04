"""Build the LoRA training set for the local-agent qwen3-coder.

Each output row is {"prompt": ..., "completion": ...} where prompt is the
fully rendered chat (system + tools + episode prefix) and completion is the
single assistant response to learn. Episodes with multiple assistant turns
yield one row per assistant turn so loss is masked to exactly the tokens the
model must produce.
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else None
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    Path("/home/asdf/Work/ailm-stack-scripts/lora_dataset.jsonl")

SYSTEM = Path("/home/asdf/.config/opencode/agents/local.md").read_text()
SYSTEM = SYSTEM.split("---", 2)[2].strip()  # strip frontmatter

TOOLS = [
    {"type": "function", "function": {
        "name": "shell",
        "description": "Run a shell command on the user's machine.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read",
        "description": "Read a file from the filesystem.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write",
        "description": "Write a file, creating directories as needed.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit",
        "description": "Apply an exact string replacement in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"}},
            "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search file contents with a regex.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "list",
        "description": "List files in a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "execute",
        "description": "Execute a call against an MCP tool server.",
        "parameters": {"type": "object", "properties": {
            "call": {"type": "string"}}, "required": ["call"]}}},
    {"type": "function", "function": {
        "name": "question",
        "description": "Ask the user a clarifying question.",
        "parameters": {"type": "object", "properties": {
            "questions": {"type": "string"}}, "required": ["questions"]}}},
    {"type": "function", "function": {
        "name": "webfetch",
        "description": "Fetch a URL and return readable text.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}}, "required": ["url"]}}},
]

GMAIL_LIST = ('/home/asdf/.local/share/gmail-mcp/venv/bin/python -c "import sys, json; '
              "sys.path.insert(0, '/home/asdf/.local/share/gmail-mcp'); import server; "
              'print(json.dumps(server.gmail_list_recent(), indent=2))"')
GMAIL_READ = ('/home/asdf/.local/share/gmail-mcp/venv/bin/python -c "import sys, json; '
              "sys.path.insert(0, '/home/asdf/.local/share/gmail-mcp'); import server; "
              'print(json.dumps(server.gmail_read(uid=\\"%s\\"), indent=2))"')
GMAIL_SEARCH = ('/home/asdf/.local/share/gmail-mcp/venv/bin/python -c "import sys, json; '
                "sys.path.insert(0, '/home/asdf/.local/share/gmail-mcp'); import server; "
                'print(json.dumps(server.gmail_search(query=\\"%s\\"), indent=2))"')


def call(name, **params):
    """Assistant tool_calls turn in OpenAI message shape; the chat template
    renders it into qwen3-coder markup."""
    return {"role": "assistant",
            "tool_calls": [{"type": "function", "function": {
                "name": name, "arguments": params}}]}


def text(s):
    return {"role": "assistant", "content": s}


def user(s):
    return {"role": "user", "content": s}


def tool(call_id, name, s):
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": s}


EMAIL_LIST_OUT = json.dumps({"messages": [
    {"uid": "32626", "mailbox": "INBOX", "from": "bdx@myworkday.com",
     "subject": "Task: BD – R-554697 Procurement & Strategic Sourcing Junior Associate - Spring 2027 (Open) – WOTC Assessment",
     "date": "Fri, 23 Oct 2026 21:36:54 +0000",
     "category": "action_required_interview_or_assessment",
     "body": "BD participates in federal and/or state tax credit programs. This assessment is voluntary and will have no impact on your application or hiring decisions. If you choose to participate, click the link to launch the assessment."},
    {"uid": "32619", "mailbox": "INBOX", "from": "Workday Notification <jj@myworkday.com>",
     "subject": "Thank you for your application for 2027 GD (U.S.) Summer Intern at Johnson & Johnson",
     "date": "Fri, 23 Oct 2026 21:10:12 +0000",
     "category": "application_status_update",
     "body": "We have successfully received your application for 2027 GD (U.S.) Summer Intern R-102105. Your application is under review, a process that usually takes up to 2 weeks."},
    {"uid": "32618", "mailbox": "INBOX", "from": "noreply@mail.amazon.jobs",
     "subject": "Amazon application: Status update",
     "date": "Fri, 23 Oct 2026 21:07:38 +0000",
     "category": "application_status_update",
     "body": "After careful consideration we have decided to progress with other candidates for Systems Development Engineer, Early Career 2026 (US) (ID: 10581943)."},
    {"uid": "32617", "mailbox": "INBOX", "from": "LinkedIn <messages-noreply@linkedin.com>",
     "subject": "Alexandra Loskutnikova is popular in your network",
     "date": "Fri, 23 Oct 2026 21:07:16 +0000",
     "category": "newsletter_or_promotion", "body": "add interesting content to your feed"},
    {"uid": "32615", "mailbox": "INBOX", "from": "info@pcrm.org",
     "subject": "Join us in November for a nutrition class series",
     "date": "Fri, 23 Oct 2026 21:01:23 +0000",
     "category": "newsletter_or_promotion", "body": "six-week online program"},
]}, indent=2)

EMAIL_SUMMARY = (
    "Here are your recent emails:\n\n"
    "- Becton Dickinson sent a voluntary WOTC assessment for the Procurement & "
    "Strategic Sourcing Junior Associate role. It is optional and does not affect hiring.\n"
    "- Johnson & Johnson received your 2027 GD Summer Intern application (R-102105); "
    "review takes up to 2 weeks.\n"
    "- Amazon rejected the Systems Development Engineer application (ID: 10581943).\n"
    "The rest are newsletters and marketing.")

LS_OUT = """total 48
drwxr-xr-x 1 asdf asdf 4096 Oct 23 10:00 .
drwx------ 1 asdf asdf 4096 Oct 23 09:00 ..
drwxr-xr-x 1 asdf asdf 4096 Oct 22 15:35 ailm-stack-scripts
drwxr-xr-x 1 asdf asdf 4096 Oct 21 20:55 job-watch
drwxr-xr-x 1 asdf asdf 4096 Oct 22 11:46 gmail_helper
drwxr-xr-x 1 asdf asdf 4096 Oct 21 22:15 SillyTavern
-rw-r--r-- 1 asdf asdf 930 Oct 21 20:55 app.log"""

CDP_OUT = json.dumps([
    {"id": "AB12", "type": "page", "title": "ChatGPT", "url": "https://chatgpt.com/"},
    {"id": "CD34", "type": "page", "title": "SillyTavern", "url": "http://localhost:8000/"},
    {"id": "EF56", "type": "page", "title": "GitHub", "url": "https://github.com/"},
], indent=2)

ST_CHATS_LS = """total 128
-rw-r--r-- 1 asdf asdf 45231 Oct 20 18:00 chat-2026-10-20.jsonl
-rw-r--r-- 1 asdf asdf 88120 Oct 22 21:14 chat-2026-10-22.jsonl"""

OPENCODE_DIR = """total 36
-rw-r--r-- 1 asdf asdf 9180 Oct 23 16:40 opencode.json
-rw-r--r-- 1 asdf asdf 2210 Oct 23 15:10 opencode.db
drwxr-xr-x 1 asdf asdf 4096 Oct 23 12:00 agents
drwxr-xr-x 1 asdf asdf 4096 Oct 23 12:00 plugins"""

FILE_NOT_FOUND = "Error: File not found: /home/asdf/.config/opencode/config.json"


def episodes():
    E = []
    cid = [0]

    def mk(msgs):
        # stamp unique tool_call ids and pair tool responses in order
        out = []
        pending = []
        for m in msgs:
            if m.get("tool_calls"):
                m = dict(m)
                m["tool_calls"] = [dict(tc, id="call_%02d" % cid[0])
                                   for tc in m["tool_calls"]]
                pending = ["call_%02d" % cid[0]] * len(m["tool_calls"])
                cid[0] += 1
            elif m["role"] == "tool" and pending:
                m = dict(m, tool_call_id=pending.pop(0))
            out.append(m)
        return out

    E.append(mk([user("check my emails and summarize them"),
                 call("shell", command=GMAIL_LIST),
                 tool("", "shell", EMAIL_LIST_OUT),
                 text(EMAIL_SUMMARY)]))

    E.append(mk([user("what's in my inbox"),
                 call("shell", command=GMAIL_LIST),
                 tool("", "shell", EMAIL_LIST_OUT),
                 text(EMAIL_SUMMARY)]))

    E.append(mk([user("did anyone get back to me about interviews?"),
                 call("shell", command=GMAIL_SEARCH % "interview OR assessment OR schedule"),
                 tool("", "shell", EMAIL_LIST_OUT),
                 text("Becton Dickinson sent a voluntary WOTC assessment for the "
                      "Procurement & Strategic Sourcing Junior Associate role. It is "
                      "optional and does not affect hiring. No interview invitations "
                      "are in this batch.")]))

    E.append(mk([user("read the most recent email fully"),
                 call("shell", command=GMAIL_LIST),
                 tool("", "shell", EMAIL_LIST_OUT),
                 call("shell", command=GMAIL_READ % "32626"),
                 tool("", "shell", "BD participates in federal and/or state tax credit "
                                   "programs. This assessment is voluntary and administered "
                                   "by a 3rd party vendor to determine the company's "
                                   "eligibility for these programs and will have no impact "
                                   "on your application, hiring decisions, retention, or "
                                   "promotion decisions."),
                 text("The newest email is Becton Dickinson's WOTC assessment for the "
                      "Procurement & Strategic Sourcing Junior Associate role. It is a "
                      "voluntary tax-credit questionnaire run by a third party; it has no "
                      "effect on your application or hiring. You can fill it via the link "
                      "in the email or ignore it.")]))

    E.append(mk([user("do you see my file system"),
                 call("glob", pattern="*"),
                 tool("", "glob", "ailm-stack-scripts/\njob-watch/\ngmail_helper/\n"
                                  "SillyTavern/\napp.log"),
                 text("Yes. The current directory holds project folders "
                      "(ailm-stack-scripts, job-watch, gmail_helper, SillyTavern) and "
                      "app.log.")]))

    E.append(mk([user("list the files in the current directory"),
                 call("shell", command="ls -la"),
                 tool("", "shell", LS_OUT),
                 text("The directory contains project folders (ailm-stack-scripts, "
                      "job-watch, gmail_helper, SillyTavern) plus app.log.")]))

    E.append(mk([user("open firefox"),
                 call("shell", command="setsid uwsm-app -- firefox >/dev/null 2>&1 &"),
                 tool("", "shell", ""),
                 text("Firefox has been opened.")]))

    E.append(mk([user("open chatgpt"),
                 call("shell", command="setsid uwsm-app -- chromium https://chatgpt.com >/dev/null 2>&1 &"),
                 tool("", "shell", ""),
                 text("ChatGPT opened in Chromium.")]))

    E.append(mk([user("what tabs are open in my browser"),
                 call("shell", command="curl -s http://[::1]:9222/json/list"),
                 tool("", "shell", CDP_OUT),
                 text("Three tabs are open in the desktop Chromium: ChatGPT, "
                      "SillyTavern (localhost:8000), and GitHub.")]))

    E.append(mk([user("delete the chats in the sillytavern webui"),
                 call("shell", command="ls ~/Work/SillyTavern/data/default-user/chats/"),
                 tool("", "shell", ST_CHATS_LS),
                 call("shell", command="rm ~/Work/SillyTavern/data/default-user/chats/chat-*.jsonl"),
                 tool("", "shell", ""),
                 text("Deleted both SillyTavern chat logs "
                      "(chat-2026-10-20.jsonl and chat-2026-10-22.jsonl).")]))

    E.append(mk([user("which browsers can you reach"),
                 call("shell", command="curl -s http://[::1]:9222/json/list"),
                 tool("", "shell", CDP_OUT),
                 call("shell", command="curl -s http://localhost:9223/json/list"),
                 tool("", "shell", "[]"),
                 text("The desktop Chromium on :9222 has three tabs (ChatGPT, "
                      "SillyTavern, GitHub). The job-watch instance on :9223 is up "
                      "with no pages.")]))

    # path-prior correction: discover the real file before editing
    E.append(mk([user("modify my opencode config so the model doesn't time out"),
                 call("shell", command="ls -la ~/.config/opencode/"),
                 tool("", "shell", OPENCODE_DIR),
                 call("read", path="/home/asdf/.config/opencode/opencode.json"),
                 tool("", "read", '{\n  "model": "ollama/qwen3-coder:30b",\n  "providers": {...}\n}'),
                 text("The config file is `~/.config/opencode/opencode.json` "
                      "(not `config.json`). It currently sets "
                      "`model: ollama/qwen3-coder:30b`. There is no timeout key in "
                      "opencode; tell me which provider or model entry you want "
                      "adjusted and I will edit it.")]))

    # fix-the-errors with a real error episode in history
    E.append(mk([user("modify my opencode config to add a timeout"),
                 call("read", path="/home/asdf/.config/opencode/config.json"),
                 tool("", "read", FILE_NOT_FOUND),
                 user("fix the erros that happened"),
                 call("shell", command="ls ~/.config/opencode/"),
                 tool("", "shell", OPENCODE_DIR),
                 call("read", path="/home/asdf/.config/opencode/opencode.json"),
                 tool("", "read", '{\n  "model": "ollama/qwen3-coder:30b",\n  "providers": {...}\n}'),
                 text("The earlier read failed because the file is named "
                      "`opencode.json`, not `config.json`. The real config uses "
                      "`model: ollama/qwen3-coder:30b`. No timeout key exists in "
                      "opencode; which provider entry should I adjust?")]))

    # error repair on a bad command
    E.append(mk([user("list my ollama models"),
                 call("shell", command="ollma list"),
                 tool("", "shell", "bash: ollma: command not found"),
                 call("shell", command="ollama list"),
                 tool("", "shell", "NAME                 ID            SIZE\n"
                                   "qwen3-coder:30b      06c1097efce0  18 GB\n"
                                   "qwen2.5:14b          188eae26bd01  9.0 GB"),
                 text("Two models are installed: qwen3-coder:30b (18 GB) and "
                      "qwen2.5:14b (9 GB).")]))

    # vague repair request WITH context must act, not ask
    E.append(mk([user("open the github cli tool output"),
                 call("shell", command="/home/asdf/.local/bin/gh pr list"),
                 tool("", "shell", "fatal: not a git repository (or any of the parent directories): .git"),
                 user("fix the error"),
                 call("shell", command="cd ~/Work/ailm-stack-scripts && /home/asdf/.local/bin/gh pr list 2>&1 | head -20"),
                 tool("", "shell", "no pull requests match your search in ailm-stack-scripts"),
                 text("The error was `gh` running outside a git repo. Rerunning "
                      "inside ~/Work/ailm-stack-scripts succeeds; there are no open "
                      "pull requests.")]))

    E.append(mk([user("summarize today's emails and open the most important link"),
                 call("shell", command=GMAIL_LIST),
                 tool("", "shell", EMAIL_LIST_OUT),
                 text("Becton Dickinson's WOTC assessment is the only actionable "
                      "item: voluntary, no effect on hiring. Amazon rejected the "
                      "Systems Development Engineer application. J&J has your intern "
                      "application under review (up to 2 weeks). The rest are "
                      "newsletters. The assessment link is inside the email body; "
                      "say the word and I will pull it up.")]))

    E.append(mk([user("find where the retry gate is defined in the shim"),
                 call("grep", pattern="can_retry_step", path="/home/asdf/Work/ailm-stack-scripts"),
                 tool("", "grep", "ailm_chat_shim.py:640:    can_retry_step = ..."),
                 text("`can_retry_step` is set in `ailm_chat_shim.py` around line 640.")]))

    E.append(mk([user("fix the typo in app.log where it says devin is down"),
                 call("read", path="/home/asdf/Work/app.log"),
                 tool("", "read", "startup ok\ndevin is down\nready"),
                 call("edit", path="/home/asdf/Work/app.log",
                      old_string="devin is down", new_string="devin is up"),
                 tool("", "edit", "Replaced 1 occurrence in /home/asdf/Work/app.log"),
                 text("Fixed: `devin is down` is now `devin is up` in app.log.")]))

    E.append(mk([user("create a script that prints disk usage"),
                 call("write", path="/home/asdf/Work/diskusage.sh",
                      content="#!/usr/bin/env bash\ndf -h"),
                 tool("", "write", "Wrote /home/asdf/Work/diskusage.sh"),
                 text("Created /home/asdf/Work/diskusage.sh.")]))

    E.append(mk([user("fetch the release notes at example.com/notes"),
                 call("webfetch", url="https://example.com/notes"),
                 tool("", "webfetch", "Release 1.2: fixes, perf."),
                 text("Release 1.2 covers fixes and performance improvements.")]))

    E.append(mk([user("send an email to bob@x.com saying the report is ready"),
                 call("shell", command=('/home/asdf/.local/share/gmail-mcp/venv/bin/python -c '
                                        '"import sys; sys.path.insert(0, \'/home/asdf/.local/share/gmail-mcp\'); '
                                        'import server; server.gmail_send(to=\\"bob@x.com\\", '
                                        'subject=\\"Report ready\\", body=\\"The report is ready.\\")"')),
                 tool("", "shell", "sent"),
                 text("Sent the email to bob@x.com.")]))

    E.append(mk([user("open github and check what tabs it opened"),
                 call("shell", command="setsid uwsm-app -- chromium https://github.com >/dev/null 2>&1 &"),
                 tool("", "shell", ""),
                 call("shell", command="curl -s http://[::1]:9222/json/list"),
                 tool("", "shell", CDP_OUT),
                 text("Chromium opened GitHub. Current tabs: ChatGPT, SillyTavern, "
                      "and GitHub.")]))

    HYPR = "HYPRLAND_INSTANCE_SIGNATURE=$(ls $XDG_RUNTIME_DIR/hypr | head -1) "

    # github goes through shell + gh, never invented execute APIs
    E.append(mk([user("check all my repos you have access to my github"),
                 call("shell", command="gh repo list --limit 30"),
                 tool("", "shell", "ailm-stack-scripts\tlocal AI stack\n"
                                   "job-watch\tjob application automation\n"
                                   "gmail_helper\tgmail helper\n"
                                   "Showing 3 of 75 repositories"),
                 text("Your GitHub has 75 repositories, including "
                      "ailm-stack-scripts, job-watch, and gmail_helper. Full "
                      "list is above.")]))

    # a failed execute method: probe Object.keys, then call a real method
    E.append(mk([user("list my github repos"),
                 {"role": "assistant", "tool_calls": [
                     {"type": "function", "function": {
                         "name": "execute",
                         "arguments": {"code": "await tools.github.list_repos()"}}}]},
                 tool("", "execute", "Error: tools.github.list_repos is not a function"),
                 {"role": "assistant", "tool_calls": [
                     {"type": "function", "function": {
                         "name": "execute",
                         "arguments": {"code": "return Object.keys(tools.github)"}}}]},
                 tool("", "execute", '["search_repositories", "search_issues", "get_file_contents", "list_pull_requests", "get_me"]'),
                 {"role": "assistant", "tool_calls": [
                     {"type": "function", "function": {
                         "name": "execute",
                         "arguments": {"code": "const r = await tools.github.search_repositories({query: 'user:barlowa124 sort:updated'}); return r.items.slice(0,10).map(x=>x.full_name)"}}}]},
                 tool("", "execute", '["barlowa124/ailm-stack-scripts", "barlowa124/job-watch"]'),
                 text("Recent repos: ailm-stack-scripts and job-watch "
                      "(via tools.github.search_repositories, after "
                      "list_repos turned out not to exist).")]))

    E.append(mk([user("check github i thinkt he keys where posted by the ojb serach caht now"),
                 call("shell", command="gh search repos \"mcp keys\" --limit 10"),
                 tool("", "shell", "no repositories matched"),
                 call("shell", command="gh search code \"api key\" --limit 10"),
                 tool("", "shell", "gh: To search code, use --owner or "
                                   "--repo to scope the query"),
                 text("No repos matched \"mcp keys\", and code search needs a "
                      "repo or owner scope. Which repo or org should I search "
                      "inside?")]))

    # window questions: hyprctl with the instance signature
    E.append(mk([user("do you see what the devin window is doing"),
                 call("shell", command=HYPR + "hyprctl clients | grep -iA3 devin"),
                 tool("", "shell", "Window 556a69fb1a40 -> Devin - Chromium:\n"
                                   "\tmapped: 1\n\tclass: chromium\n"
                                   "\ttitle: Devin - Chromium"),
                 call("shell", command="devin-status | head -20"),
                 tool("", "shell", "agate-astronaut\tfree\tContinue Interrupted Response\n"
                                   "sturdy-pegasus\tfree\tContinue Job Search Work"),
                 text("The Devin window is a Chromium window titled "
                      "\"Devin - Chromium\". `devin-status` shows two open "
                      "sessions: \"Continue Interrupted Response\" and "
                      "\"Continue Job Search Work\", both idle.")]))

    E.append(mk([user("what is the firefox window doing"),
                 call("shell", command=HYPR + "hyprctl clients"),
                 tool("", "shell", "Window 556a69fb1c10 -> YouTube - Mozilla Firefox:\n"
                                   "\tfloating: 0\n\tworkspace: 2\n"
                                   "\tclass: firefox\n\ttitle: YouTube - Mozilla Firefox"),
                 text("Firefox has one mapped window on workspace 2 playing "
                      "YouTube.")]))

    E.append(mk([user("how do i close a window on omarchy"),
                 call("shell", command="grep -niE 'close|killactive' ~/.config/hypr/bindings.lua"),
                 tool("", "shell", "o.bind(\"SUPER + W\", \"Close window\", smart_close)\n"
                                   "o.bind(\"SUPER + Q\", \"Close window\", smart_close)\n"
                                   "o.bind(\"SUPER + SHIFT + Q\", \"Force kill window\", hl.dsp.window.kill())"),
                 text("SUPER+Q or SUPER+W closes a window. SUPER+SHIFT+Q "
                      "force-kills one that ignores the close request.")]))

    # omarchy idle control uses `omarchy toggle idle`, never invented power scripts
    E.append(mk([user("disable omarchy from falling asleep"),
                 call("shell", command="omarchy toggle idle stay-awake"),
                 tool("", "shell", "disabled"),
                 text("Automatic sleep is now disabled (`omarchy toggle idle stay-awake`). "
                      "Run `omarchy toggle idle allow-idle` to re-enable it.")]))

    E.append(mk([user("is my computer set to stay awake or will it sleep"),
                 call("shell", command="omarchy toggle idle status"),
                 tool("", "shell", '{"enabled":false,"class":"disabled","tooltip":"Stay Awake"}'),
                 text("Idle sleep is currently allowed (`\"enabled\":false` means "
                      "stay-awake is off). Run `omarchy toggle idle stay-awake` to "
                      "keep the machine awake.")]))

    E.append(mk([user("where is stremio i want ot watch a movie"),
                 call("shell", command="which stremio 2>/dev/null; flatpak list 2>/dev/null | grep -i stremio; ls /usr/bin | grep -i strem"),
                 tool("", "shell", ""),
                 text("Stremio is not installed (not on PATH, not a flatpak). "
                      "Want me to install it with `omarchy pkg add stremio`?")]))

    E.append(mk([user("check my newegg for what the speakers are"),
                 call("shell", command="curl -s http://[::1]:9222/json/list | grep -i newegg"),
                 tool("", "shell", "{\"id\":\"F1\",\"type\":\"page\",\"title\":\"Newegg.com - Order History\",\"url\":\"https://www.newegg.com/account/orders\"}"),
                 text("You have a Newegg order-history tab open in the desktop "
                      "Chromium (:9222). I can read the order details from that "
                      "tab via CDP if you want the speaker model.")]))

    E.append(mk([user("a process crashed, quickshell SIGSEGV pid 12345, find out why"),
                 call("shell", command="coredumpctl info 12345 2>/dev/null | head -40"),
                 tool("", "shell", "PID: 12345 (quickshell)\nSignal: 11 (SIGSEGV)\n"
                                   "Command line: /usr/bin/quickshell\n"
                                   "Stack trace of thread 12345:\n#0 qs::ipc::IpcServer::onNewConnection"),
                 call("shell", command="journalctl -b -p err --no-pager | tail -20"),
                 tool("", "shell", "Oct 23 quickshell[12345]: segfault at 8 ip libQt6.so"),
                 text("quickshell segfaulted (SIGSEGV) inside "
                      "qs::ipc::IpcServer::onNewConnection per the coredumpctl "
                      "backtrace; journal shows a null-ish read at address 8 in "
                      "libQt6. That points at an IPC client disconnecting mid-"
                      "message, a known crash class, not a config error.")]))

    E.append(mk([user("what are my emails today"),
                 call("shell", command=GMAIL_LIST),
                 tool("", "shell", EMAIL_LIST_OUT),
                 text(EMAIL_SUMMARY)]))

    E.append(mk([user("what is the output of running date in bash"),
                 call("shell", command="date"),
                 tool("", "shell", "Fri Oct 23 19:30:12 EDT 2026"),
                 text("`date` printed: Fri Oct 23 19:30:12 EDT 2026.")]))

    # drives + desktop setup continues with real actions, not invented APIs
    E.append(mk([user("check my drives and set up my desktop"),
                 call("shell", command="lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT"),
                 tool("", "shell", "sda   931G\n└─sda2  931G ntfs /mnt/sda2\n"
                                   "nvme0n1 1.9T\n└─nvme0n1p2 1.9T ext4 /"),
                 call("shell", command="omarchy commands | head -40"),
                 tool("", "shell", "omarchy theme set <name>\n"
                                   "omarchy capture screenshot\n"
                                   "omarchy restart hypridle"),
                 text("Drives look right: / on nvme0n1p2 plus data mounts under "
                      "/mnt. For the desktop, `omarchy` covers theme, bar, and "
                      "idle settings. Which piece first: theme, wallpaper, or "
                      "monitor layout?")]))

    # desktop CDP is checked before job-watch CDP
    E.append(mk([user("is there an amazon tab open"),
                 call("shell", command="curl -s http://[::1]:9222/json/list | grep -i amazon"),
                 tool("", "shell", "{\"id\":\"A9\",\"type\":\"page\",\"title\":\"Your Orders\",\"url\":\"https://www.amazon.com/gp/css/order-history\"}"),
                 text("Yes, the desktop Chromium has an Amazon order-history "
                      "tab open.")]))

    E.append(mk([user("check my newegg for what the speakers are"),
                 call("shell", command="curl -s http://[::1]:9222/json/list | grep -i newegg"),
                 tool("", "shell", ""),
                 call("shell", command="curl -s http://localhost:9223/json/list | grep -i newegg"),
                 tool("", "shell", ""),
                 text("No Newegg tab is open in either Chromium instance "
                      "(:9222 desktop or :9223 job-watch). Want me to open "
                      "newegg.com in the desktop browser?")]))

    # opencode integration targets real config surfaces, not invented dirs
    E.append(mk([user("integrate devin with opencode"),
                 call("shell", command="ls ~/.config/opencode/"),
                 tool("", "shell", OPENCODE_DIR),
                 call("read", path="/home/asdf/.config/opencode/opencode.json"),
                 tool("", "read", '{\n  "model": "ollama/qwen3-coder:30b",\n'
                                   '  "provider": {"devin": {"models": {'
                                   '"swe-2-max": {}, "gemini-4-argon-high": {}}},\n'
                                   '  "providers": {...}\n}'),
                 call("shell", command="which devin && devin --version"),
                 tool("", "shell", "/home/asdf/.local/bin/devin\n3000.11.205"),
                 text("Devin is already integrated: `devin` CLI 3000.11.205 is "
                      "on PATH and opencode.json has a `devin` provider with "
                      "swe-2-max and gemini-4-argon-high. To use one as an "
                      "agent, reference it as `devin/swe-2-max` in a request "
                      "or set it as `model` in an agent file under "
                      "~/.config/opencode/agents/.")]))

    # searches broaden before reporting failure, no invented orgs
    E.append(mk([user("check github i thinkt he keys where posted by the ojb serach caht now"),
                 call("shell", command="gh search code \"api_key\" --owner barlowa124 --limit 10"),
                 tool("", "shell", "no code matched \"api_key\" in owner:barlowa124"),
                 call("shell", command="gh gist list --limit 10"),
                 tool("", "shell", "mcp-setup\tkeys for mcp servers\t1 file"),
                 call("shell", command="gh repo list --limit 10 | grep -iE 'job|search|mcp'"),
                 tool("", "shell", "barlowa124/job-watch\tJob-board monitoring\n"
                                   "barlowa124/omarchy-job-search\tjob sweep context"),
                 text("No `api_key` hits in your repos, but the gist "
                      "\"mcp-setup\" and the omarchy-job-search repo both look "
                      "like candidates for posted keys. Want me to read the "
                      "mcp-setup gist?")]))

    # installed-app checks are exhaustive before concluding
    E.append(mk([user("where is stremio i want ot watch a movie"),
                 call("shell", command="which stremio; flatpak list 2>/dev/null | grep -i stremio; ls ~/.local/share/applications | grep -i stremio; pacman -Qs stremio"),
                 tool("", "shell", ""),
                 call("shell", command="ls ~/Work | grep -i strem"),
                 tool("", "shell", "stremio-local-torrentio"),
                 text("Stremio is not installed system-wide, but "
                      "~/Work/stremio-local-torrentio exists, a local setup "
                      "project for it. Want me to check that directory or "
                      "install stremio with `omarchy pkg add`?")]))

    E.append(mk([user("did you install devin"),
                 call("shell", command="which devin && devin --version"),
                 tool("", "shell", "/home/asdf/.local/bin/devin\n3000.11.205"),
                 text("Yes, the Devin CLI is installed at ~/.local/bin/devin "
                      "(version 3000.11.205).")]))

    E.append(mk([user("open devin"),
                 call("shell", command="ls /usr/share/applications ~/.local/share/applications 2>/dev/null | grep -i devin"),
                 tool("", "shell", "devin.desktop"),
                 call("shell", command="setsid uwsm-app -- gtk-launch devin >/dev/null 2>&1 &"),
                 tool("", "shell", ""),
                 text("Launched Devin from its desktop entry.")]))

    E.append(mk([user("i want to swtich models to swe max and gemini argon"),
                 call("read", path="/home/asdf/.config/opencode/opencode.json"),
                 tool("", "read", '{\n  "model": "ollama/qwen3-coder:30b",\n'
                                   '  "provider": {"devin": {"models": {'
                                   '"swe-2-max": {"name": "SWE-2 Max"}, '
                                   '"gemini-4-argon-high": {"name": "Gemini Argon"}}},\n'
                                   '  "providers": {...}\n}'),
                 text("Both are already configured under the `devin` provider: "
                      "`devin/swe-2-max` and `devin/gemini-4-argon-high`. Pick "
                      "one in the model switcher, or say which and I will set "
                      "it in opencode.json.")]))

    # calibrated question use: genuinely ambiguous, two live options
    E.append(mk([user("should I use the 32b or keep the 14b as my daily"),
                 call("question", questions=("Which should I set as the daily model? "
                                             "qwen3-coder:30b (stronger, 18GB) or "
                                             "qwen2.5:14b (lighter, 9GB)")),
                 tool("", "question", "The user will answer in chat."),
                 text("Pick one in the question prompt and I will update the "
                      "config.")]))

    # recovery: agents.d does not exist -> pivot to the real config surface
    E.append(mk([user("set up devin inside opencode"),
                 call("shell", command="ls ~/.config/opencode/agents.d/"),
                 tool("", "shell", "ls: cannot access '/home/asdf/.config/opencode/agents.d/': No such file or directory"),
                 call("shell", command="ls ~/.config/opencode/"),
                 tool("", "shell", OPENCODE_DIR),
                 call("read", path="/home/asdf/.config/opencode/opencode.json"),
                 tool("", "read", '{\n  "model": "ollama/qwen3-coder:30b",\n'
                                   '  "provider": {"devin": {"models": {'
                                   '"swe-2-max": {}, "gemini-4-argon-high": {}}},\n'
                                   '  "providers": {...}\n}'),
                 text("`agents.d` is not a real opencode path; agents live in "
                      "`~/.config/opencode/agents/` and models in `opencode.json`. "
                      "The devin provider already lists swe-2-max and "
                      "gemini-4-argon-high, so devin is usable as "
                      "`devin/swe-2-max` in any agent.")]))

    E.append(mk([user("hook devin into opencode"),
                 call("shell", command="which devin && devin --version"),
                 tool("", "shell", "/home/asdf/.local/bin/devin\n3000.11.205"),
                 call("shell", command="ls ~/.config/opencode/agents/"),
                 tool("", "shell", "local.md"),
                 call("read", path="/home/asdf/.config/opencode/opencode.json"),
                 tool("", "read", '{\n  "model": "ollama/qwen3-coder:30b",\n'
                                   '  "provider": {"devin": {"models": {'
                                   '"swe-2-max": {}, "gemini-4-argon-high": {}}},\n'
                                   '  "providers": {...}\n}'),
                 text("Devin CLI 3000.11.205 is installed and opencode.json "
                      "already has a `devin` provider with swe-2-max and "
                      "gemini-4-argon-high. Agents in ~/.config/opencode/agents/ "
                      "can select them via `model: devin/<name>`. Nothing to "
                      "install; which agent should use it?")]))


    # invented execute namespace for opencode -> recover via shell/read
    E.append(mk([user("show my opencode worktrees"),
                 {"role": "assistant", "tool_calls": [
                     {"type": "function", "function": {
                         "name": "execute",
                         "arguments": {"code": "await tools.opencode.worktree_list()"}}}]},
                 tool("", "execute", "Error: tools is not defined"),
                 call("shell", command="ls ~/.config/opencode/ && git -C ~/Work worktree list 2>/dev/null | head"),
                 tool("", "shell", OPENCODE_DIR + "\n/home/asdf/Work  abc1234 [main]"),
                 text("`execute` has no opencode namespace, so I checked the "
                      "config dir and git directly: opencode config lives in "
                      "~/.config/opencode and ~/Work has one worktree on "
                      "main. There is no opencode worktree feature to list.")]))


    # stremio paraphrases: broaden a failed which
    E.append(mk([user("open stremio"),
                 call("shell", command="which stremio"),
                 tool("", "shell", "which: no stremio in (/home/asdf/.local/bin:...)"),
                 call("shell", command="flatpak list 2>/dev/null | grep -i stremio; ls /usr/share/applications ~/.local/share/applications 2>/dev/null | grep -i strem; pacman -Qs stremio 2>/dev/null; ls ~/Work | grep -i strem"),
                 tool("", "shell", "stremio-local-torrentio"),
                 text("Stremio is not installed system-wide, but "
                      "~/Work/stremio-local-torrentio exists. Want me to check "
                      "that project or install stremio with `omarchy pkg add`?")]))

    E.append(mk([user("where is spotify installed"),
                 call("shell", command="which spotify; flatpak list 2>/dev/null | grep -i spotify; ls /usr/share/applications ~/.local/share/applications 2>/dev/null | grep -i spotify; pacman -Qs spotify 2>/dev/null"),
                 tool("", "shell", "com.spotify.Client"),
                 text("Spotify is installed as the flatpak `com.spotify.Client`. "
                      "Launch it with `flatpak run com.spotify.Client` or its "
                      "desktop entry.")]))


    # a date/trip question hunts bookings in gmail and calendar, not ~/Work
    E.append(mk([user("when am i doing the thanksgiving trip to see my parents, i want a non insane price and ticket"),
                 call("shell", command=GMAIL_SEARCH % "flight OR itinerary OR booking OR airline"),
                 tool("", "shell", json.dumps({"messages": [
                     {"uid": "32590", "from": "confirm@aa.com",
                      "subject": "Your trip confirmation - BOS",
                      "date": "Wed, 15 Oct 2026 10:00:00 +0000",
                      "body": "Boston round trip, depart Nov 25, return Nov 30, confirmation XYZ123, $412"}]}, indent=2)),
                 call("shell", command="ls ~/.config/omarchy ~/.local/share/calcurse 2>/dev/null; khal list 2>/dev/null | grep -iE 'trip|travel|parent' | head"),
                 tool("", "shell", ""),
                 text("Gmail has an American Airlines confirmation: Boston "
                      "round trip departing Nov 25, returning Nov 30, "
                      "confirmation XYZ123 at $412. No calendar entries "
                      "mention the trip. For a cheaper fare, the Firefox "
                      "window already has Google Flights open on a Boston "
                      "round trip; I can check prices there via CDP.")]))


    # desktop icons: hyprland has no icon surface, report honestly
    E.append(mk([user("set up my desktop icons base don what i have"),
                 call("shell", command="ls ~/Desktop 2>/dev/null; ls /usr/share/applications | wc -l; grep -niE 'icon|desktop' ~/.config/hypr/hyprland.lua | head"),
                 tool("", "shell", "214"),
                 text("Hyprland has no desktop-icon surface; ~/Desktop is "
                      "empty and nothing in hyprland.lua renders icons. There "
                      "are 214 installed .desktop entries. The practical "
                      "setup here is the launcher (SUPER+Space) or waybar "
                      "pinned apps. Want me to pin your most-used apps to "
                      "waybar instead?")]))


    # model switching: read config first, do not loop devin-status
    E.append(mk([user("switch to argon"),
                 call("read", path="/home/asdf/.config/opencode/opencode.json"),
                 tool("", "read", '{\n  "model": "ollama/qwen3-coder:30b",\n'
                                   '  "provider": {"devin": {"models": {'
                                   '"swe-2-max": {}, "gemini-4-argon-high": {}}},\n'
                                   '  "providers": {...}\n}'),
                 call("edit", path="/home/asdf/.config/opencode/opencode.json",
                      old_string='"model": "ollama/qwen3-coder:30b"',
                      new_string='"model": "devin/gemini-4-argon-high"'),
                 tool("", "edit", "Replaced 1 occurrence in opencode.json"),
                 text("Switched the opencode default to "
                      "`devin/gemini-4-argon-high` in opencode.json. New "
                      "sessions use Argon; say the word to switch back.")]))


    # newegg recovery: :9222 empty -> check :9223 before concluding
    E.append(mk([user("check my newegg for what the speakers are"),
                 call("shell", command="curl -s http://[::1]:9222/json/list | grep -i newegg"),
                 tool("", "shell", ""),
                 call("shell", command="curl -s http://localhost:9223/json/list | grep -i newegg"),
                 tool("", "shell", "{\"id\":\"J7\",\"type\":\"page\",\"title\":\"Newegg.com - Shopping Cart\",\"url\":\"https://www.newegg.com/cart\"}"),
                 text("No Newegg tab in the desktop Chromium, but the "
                      "job-watch Chromium (:9223) has a Newegg shopping-cart "
                      "tab open. I can query that tab's page over CDP to "
                      "read the speaker listing.")]))


    # job-search state lives in ~/Work/job-watch, not gmail
    E.append(mk([user("what interviews do i have coming up"),
                 call("shell", command="tail -40 ~/Work/job-watch/replies.log"),
                 tool("", "shell", "2026-10-21 Kalmus Labs: booked Sat Oct 24 12:30-13:00 Google Meet meet.google.com/vxr-ywcx-mpo\n"
                                   "2026-10-23 Cencora: rescheduled to Mon Oct 26 10:30-10:45 ET MS Teams\n"
                                   "2026-10-22 Regeneron: Modern Hire invite received, not booked"),
                 call("shell", command="grep -niE 'interview|meet|scheduled' ~/Work/job-watch/applications_log.md | tail -10"),
                 tool("", "shell", "Kalmus Labs. Founding Backend & AI Systems Engineer\n- Interview booked Sat Oct 24 12:30"),
                 text("Two interviews are booked. Kalmus Labs (Remora) today "
                      "Sat Oct 24 at 12:30-1:00 PM on Google Meet "
                      "(meet.google.com/vxr-ywcx-mpo), and Cencora Mon Oct 26 "
                      "at 10:30-10:45 AM ET on MS Teams. Regeneron sent a "
                      "Modern Hire invite that is not yet booked.")]))


    E.append(mk([user("how many jbos have been applied to at this point"),
                 call("shell", command="grep -c SUBMITTED ~/Work/job-watch/applications_log.txt && grep -c BLOCKED ~/Work/job-watch/applications_log.txt"),
                 tool("", "shell", "637\n41"),
                 text("637 applications have been submitted so far (with 41 "
                      "blocked entries in `applications_log.txt`).")]))


    # meeting links: gmail search + calendar + desktop notes
    E.append(mk([user("where is the zoom link"),
                 call("shell", command=GMAIL_SEARCH % "zoom OR meet.google.com OR interview"),
                 tool("", "shell", json.dumps({"messages": [
                     {"uid": "32680", "from": "admin@kalmuslabs.com",
                      "subject": "Reminder: Remora - Candidate Interview",
                      "date": "Fri, 23 Oct 2026 18:00:00 +0000",
                      "body": "Reminder of your interview Sat Oct 24 12:30-1:00 PM EDT. Join: https://meet.google.com/vxr-ywcx-mpo Dial-in: +1 636-498-4573 PIN 291750901"}]}, indent=2)),
                 text("There is no Zoom link; today's interview is on Google "
                      "Meet. The Kalmus Labs (Remora) candidate interview is "
                      "Sat Oct 24, 12:30-1:00 PM EDT at "
                      "https://meet.google.com/vxr-ywcx-mpo (dial-in "
                      "+1 636-498-4573, PIN 291750901), per the reminder email "
                      "from admin@kalmuslabs.com.")]))


    # multi-part: emails AND github both get used
    E.append(mk([user("check my rejections in my emails how can i make my github stack more appealing to thsese positions"),
                 call("shell", command=GMAIL_SEARCH % "rejected OR not moving forward OR other candidates"),
                 tool("", "shell", json.dumps({"messages": [
                     {"uid": "32749", "from": "bdx@myworkday.com",
                      "subject": "BD Job Application Status",
                      "body": "we will not be moving forward with your application for Software Engineer I"},
                     {"uid": "32730", "from": "careers@cencora.com",
                      "subject": "Application update",
                      "body": "you were not selected to move forward in the recruiting process"}]}, indent=2)),
                 call("shell", command="gh repo list --limit 30 --json name,description,isArchived,pushedAt --jq '.[] | \"\\(.name)\\t\\(.isArchived)\\t\\(.pushedAt)\"' | head -20"),
                 tool("", "shell", "omarchy-job-search\tfalse\t2026-10-21\n"
                                   "protein-ml\tfalse\t2026-10-20\n"
                                   "job-watch-private\tfalse\t2026-10-21\n"
                                   "cytof-qc\tfalse\t2026-08-11"),
                 call("shell", command="gh search prs --author=barlowa124 is:merged --limit 20 --json repository,title --jq '.[].repository' | sort | uniq -c | sort -rn"),
                 tool("", "shell", "5 rdkit/rdkit\n4 zarr-developers/zarr-python\n3 MDAnalysis/mdanalysis"),
                 text("Recent rejections: BD (Software Engineer I) and Cencora "
                      "(Finance Co-Op), both generic 'not moving forward' "
                      "notices. For the github side: you have 12 merged PRs "
                      "across rdkit, zarr-python, and MDAnalysis, but the "
                      "profile shows them nowhere. Add an 'Open-source "
                      "contributions' section to your profile README listing "
                      "the merged PRs, and archive stale repos like cytof-qc "
                      "that have not moved since August.")]))


    # github state -> gh, never invented execute namespaces
    E.append(mk([user("check the current state of the github there are many prs and repos that work was done on"),
                 call("shell", command="gh search prs --author=barlowa124 is:open --limit 30 --json repository,title,number --jq '.[] | \"\\(.repository)\\t#\\(.number)\\t\\(.title)\"' | head -20"),
                 tool("", "shell", "rdkit/rdkit\t#9671\tfix descriptor cache\n"
                                   "zarr-developers/zarr-python\t#4485\tcodec fix"),
                 call("shell", command="gh search prs --author=barlowa124 is:closed is:unmerged --limit 15 --json repository,number --jq '.[].repository' | sort | uniq -c | sort -rn"),
                 tool("", "shell", "5 OpenADMET/openadmet-models\n2 jelmer/dulwich\n1 MDAnalysis/mdanalysis"),
                 call("shell", command="gh repo list --limit 15 --json name,pushedAt --jq '.[] | \"\\(.name)\\t\\(.pushedAt)\"'"),
                 tool("", "shell", "omarchy-job-search\t2026-10-21\njob-watch-private\t2026-10-21\nailm-stack-scripts\t2026-10-23"),
                 text("GitHub state: about 60 PRs open including rdkit#9671 "
                      "and zarr-python#4485, 15 closed-unmerged mostly in "
                      "OpenADMET and dulwich, and recent pushes to "
                      "omarchy-job-search, job-watch-private, and "
                      "ailm-stack-scripts.")]))


    # invented method on a real namespace -> probe Object.keys, then succeed
    E.append(mk([user("what does my github profile look like"),
                 {"role": "assistant", "tool_calls": [
                     {"type": "function", "function": {
                         "name": "execute",
                         "arguments": {"code": "await tools.github.whoami()"}}}]},
                 tool("", "execute", "Error: tools.github.whoami is not a function"),
                 {"role": "assistant", "tool_calls": [
                     {"type": "function", "function": {
                         "name": "execute",
                         "arguments": {"code": "return Object.keys(tools.github)"}}}]},
                 tool("", "execute", '["search_repositories", "search_issues", "get_me", "list_pull_requests"]'),
                 {"role": "assistant", "tool_calls": [
                     {"type": "function", "function": {
                         "name": "execute",
                         "arguments": {"code": "return await tools.github.get_me()"}}}]},
                 tool("", "execute", '{"login": "barlowa124", "public_repos": 75, "followers": 3}'),
                 text("Profile barlowa124: 75 public repos, 3 followers "
                      "(via tools.github.get_me).")]))


    # deep verification before declaring setup done
    E.append(mk([user("check my omarchy installation, i want ot download devin sign into devin so it can set up my computer i just installed omarchy"),
                 call("shell", command="pacman -Q omarchy devin-desktop 2>/dev/null; omarchy --version 2>/dev/null | head -2; systemctl --failed --no-legend | head"),
                 tool("", "shell", "omarchy 4.0.6-1\ndevin-desktop 3.10.67-1\nomarchy 4.0.6"),
                 call("shell", command="which devin && devin --version && pgrep -af devin | head -5"),
                 tool("", "shell", "/home/asdf/.local/bin/devin\n3000.11.205\n"),
                 call("shell", command="setsid uwsm-app -- devin-desktop >/dev/null 2>&1 &"),
                 tool("", "shell", ""),
                 text("Everything is already in place. Omarchy 4.0.6 is "
                      "installed with no failed services, devin-desktop "
                      "3.10.67 is installed as a package, and the devin CLI "
                      "3000.11.205 is on PATH and authenticated (this session "
                      "runs through it). I launched Devin Desktop; it will "
                      "show your existing sessions if the login is still "
                      "cached, otherwise a sign-in screen.")]))


    # peripheral entity tracking: report when the asked-about bud is absent
    E.append(mk([user("the left bud is not out of case it is in case"),
                 call("shell", command="bluetoothctl devices && bluetoothctl devices Connected"),
                 tool("", "shell", "Device 80:99:E7:FE:AD:96 WH-1000XM6\n"
                                   "Device 80:99:E7:FE:AD:96 WH-1000XM6"),
                 text("`bluetoothctl` only lists the Sony WH-1000XM6 over-ear "
                      "headset (currently connected). No earbuds appear in "
                      "the paired or connected Bluetooth device list, and "
                      "BlueZ does not report per-earbud case state.")]))

    E.append(mk([user("what is the battery level on my airpods"),
                 call("shell", command="bluetoothctl devices && bluetoothctl devices Connected"),
                 tool("", "shell", "Device 80:99:E7:FE:AD:96 WH-1000XM6\n"
                                   "Device 80:99:E7:FE:AD:96 WH-1000XM6"),
                 text("No AirPods are paired on this machine. The only paired "
                      "and connected Bluetooth audio device is `WH-1000XM6` "
                      "(80:99:E7:FE:AD:96).")]))

    # launcher-to-game window matching: Steam games appear under the game title
    E.append(mk([user("i cant move the window that contains steam and resize it"),
                 call("shell", command=HYPR + "hyprctl clients | grep -E 'Window |class:|title:|floating:|fullscreen:'"),
                 tool("", "shell", "Window 556a69fb1d00 -> Skyrim Special Edition:\n"
                                   "\tfloating: 0\n\tfullscreen: 0\n"
                                   "\tclass: steam_app_489830\n"
                                   "\ttitle: Skyrim Special Edition\n"
                                   "Window 556a69fb1e20 -> Darkest Dungeon:\n"
                                   "\tfloating: 0\n\tfullscreen: 0\n"
                                   "\tclass: Darkest.bin.x86_64\n"
                                   "\ttitle: Darkest Dungeon"),
                 call("shell", command="grep -nA8 'steam_app_\\|mouse:272' ~/.config/hypr/hyprland.lua ~/.config/hypr/bindings.lua"),
                 tool("", "shell", "/home/asdf/.config/hypr/hyprland.lua:99:o.window(\"(\\\\.exe$|^steam_app_)\", {\n"
                                   "/home/asdf/.config/hypr/hyprland.lua:100:  tile = true,\n"
                                   "/home/asdf/.config/hypr/bindings.lua:32:o.bind(\"ALT + mouse:272\", \"Move window (Alt)\", hl.dsp.window.drag(), { mouse = true })\n"
                                   "/home/asdf/.config/hypr/bindings.lua:33:o.bind(\"ALT + mouse:273\", \"Resize window (Alt)\", hl.dsp.window.resize(), { mouse = true })"),
                 text("No standalone Steam client window is open, only the "
                      "Steam game windows (`Skyrim Special Edition` with class "
                      "`steam_app_489830`, and `Darkest Dungeon`). Both are "
                      "tiled (`floating: 0`) because `hyprland.lua` line 99 "
                      "forces `tile = true` on `(\\.exe$|^steam_app_)`. To "
                      "move or resize one freely, float it first with "
                      "`SUPER+T` (or `hyprctl dispatch togglefloating`), then "
                      "drag with `SUPER + LMB/RMB` or `ALT + LMB/RMB`.")]))

    E.append(mk([user("where is the lutris window"),
                 call("shell", command=HYPR + "hyprctl clients | grep -E 'Window |class:|title:|workspace:'"),
                 tool("", "shell", "Window 556a69fb2a10 -> Battle.net:\n"
                                   "\tworkspace: 3 (3)\n"
                                   "\tclass: wine64-preloader\n"
                                   "\ttitle: Battle.net"),
                 text("No Lutris launcher window is mapped, but the Wine child "
                      "window `Battle.net` (`class: wine64-preloader`) is open "
                      "on workspace 3.")]))

    # current-prompt references: multi-provider quota, Harbor F8 media key, and Ollama residency
    E.append(mk([user("i need to see my devin quota to tell when the creidts are about to be exhausted or im on over limit credits in opencode"),
                 call("shell", command="/home/asdf/.local/bin/devin-quota"),
                 tool("", "shell", "=== 1. Multi-Provider Quota & Empirical Burn Trajectory (Raw vs. Fusion-Adjusted) ===\n"
                                   "  [Devin]      Devin Max [RAW PACE 5h 6m (30.14x) -> FUSION 17h 57m (8.58x, +12h 50m paid runway, 46% left)]\n"
                                   "               Weekly: 46% left (54% used in 13.85h · Sustainable: 0.3%/h · Reset: 2026-11-01 03:00 in 6d 10h)\n"
                                   "               Daily: 100% left (resets 2026-10-26 04:00, in 10h 9m) · Over-limit: $-0.06 [EXHAUSTED]\n"
                                   "  [ChatGPT Plus (OpenCode OAuth) (Included in Plus ($0 Devin quota))] Connected (barlowa124@gmail.com · openai/gpt-6.1-sol)\n"
                                   "  [Google One AI Pro / Gemini API (Google AI Pro ($0 Devin quota))] Connected (barlowa124@gmail.com · 32 google/gemini-* models)\n"
                                   "  [OpenCode Zen (Built-in Free Pool) (Free ($0.00, no key needed))] 8 live free models"),
                 text("Devin Max has 46% weekly quota remaining (resets 2026-11-01 03:00) "
                      "and 100% daily quota remaining, with over-limit balance at -$0.06 "
                      "(exhausted). Raw burn pace is 30.14x (5h 6m to exhaustion), while "
                      "Custom Fusion Mode reduces burn by 71.5% to 8.58x (17h 57m paid "
                      "runway, +12h 50m gained) before automatically switching to the $0.00 "
                      "free-mix (swe-2-max Free Promo + local qwen3-coder:30b-ft3 + Gemini "
                      "verifier). ChatGPT Plus OAuth, Google Gemini API, and 8 OpenCode Zen "
                      "free models are also active.")]))

    E.append(mk([user("harrbvor is widescreen im not sure how i can pause f8 to pause/play on my keyboard isnt owkring"),
                 call("shell", command="grep -n 'harbor-media\\|XF86AudioPlay\\|F8' ~/.config/hypr/bindings.lua"),
                 tool("", "shell", "45:o.bind(\"XF86AudioPlay\", \"Play/Pause media\", hl.dsp.exec(\"~/.local/bin/harbor-media playpause\"), { locked = true })\n"
                                   "47:o.bind(\"F8\", \"Play/Pause media (F8)\", hl.dsp.exec(\"~/.local/bin/harbor-media playpause\"), { locked = true })"),
                 text("`F8` and `XF86AudioPlay` are bound in `~/.config/hypr/bindings.lua` "
                      "to `~/.local/bin/harbor-media playpause`, which invokes Harbor's "
                      "Play/Pause button over AT-SPI and falls back to `omarchy-shell media` "
                      "for other players. To toggle fullscreen on the Harbor window, focus "
                      "it and press `SUPER + F`.")]))

    E.append(mk([user("i dont have any chats currently using 30b coder why i it still mounted"),
                 call("shell", command="curl -s http://127.0.0.1:11434/api/ps && echo '' && curl -s http://127.0.0.1:8015/health"),
                 tool("", "shell", '{"models":[]}\n{"ok": true, "provider": "remote", "mode": "http://127.0.0.1:11434/v1"}'),
                 text("`ollama/ps` shows no models currently loaded in VRAM (`models: []`). "
                      "The `:8015` chat shim automatically unloads Ollama models when all "
                      "active OpenCode sessions switch away from the `ollama` provider or "
                      "after the 10-minute idle grace window.")]))

    # Pass-3 flow trajectories: pre-tool continuation inheritance and follow-up context
    E.append(mk([user("the bluetooth isnt connecting to the headphones the bluetooth process is crashing"),
                 call("shell", command="bluetoothctl devices && bluetoothctl info 80:99:E7:FE:AD:96"),
                 tool("", "shell", "Device 80:99:E7:FE:AD:96 WH-1000XM6\n\tName: WH-1000XM6\n\tPaired: yes\n\tConnected: yes"),
                 text("`WH-1000XM6` (`80:99:E7:FE:AD:96`) is paired and currently connected."),
                 user("[Active Task Reference: the bluetooth isnt connecting to the headphones the bluetooth process is crashing]\n\nconitnue"),
                 call("shell", command="coredumpctl list bluetoothctl bluetoothd 2>/dev/null | tail -5; journalctl -u bluetooth -n 20 --no-pager"),
                 tool("", "shell", "Sun 2026-10-25 14:12:01 EDT 418920 1000 1000 SIGABRT present /usr/bin/bluetoothctl\n"
                                   "Oct 25 14:12:01 omarchy bluetoothd[912]: Endpoint registered: sender=:1.44 path=/MediaEndpoint/A2DPSource/sbc"),
                 text("`bluetoothd` is active with the SBC A2DP endpoint registered for "
                      "`WH-1000XM6`, while `coredumpctl` recorded a `SIGABRT` in the "
                      "`/usr/bin/bluetoothctl` CLI client (PID 418920) rather than the "
                      "daemon.")]))

    E.append(mk([user("delete the chats in the chatgpt webui"),
                 call("shell", command="curl -s http://[::1]:9222/json/list"),
                 tool("", "shell", CDP_OUT),
                 text("Desktop Chromium on `:9222` has an active ChatGPT tab (`https://chatgpt.com/`, id `AB12`)."),
                 user("[Active Task Reference: delete the chats in the chatgpt webui]\n\nyou have chromium"),
                 call("shell", command="curl -s http://[::1]:9222/json/list"),
                 tool("", "shell", CDP_OUT),
                 text("Found the open ChatGPT tab (`id: AB12`, `https://chatgpt.com/`) on "
                      "desktop Chromium CDP `:9222`.")]))

    E.append(mk([user("[Active Task Reference: create a custom fusion mode that combines the paid and free models]\n\ndoa third pass use current prompts as reference"),
                 call("shell", command="/home/asdf/.local/bin/devin-quota --short && curl -s http://127.0.0.1:8015/health && echo '' && curl -s http://127.0.0.1:41435/fusion"),
                 tool("", "shell", "Devin Max: Weekly 45% (Raw 9.17%/h -> Fusion 2.28%/h, exhausts in 19h 44m [+14h 50m vs 4h 54m raw, 4.03x] + $0 free-mix) · Daily 100% · Extra $-0.06\n"
                                   "{\"ok\": true, \"provider\": \"remote\", \"mode\": \"http://127.0.0.1:11434/v1\", \"mixture\": {\"fusion_mode\": \"active\", \"turns_verified\": 4, \"turns_corrected\": 2}}\n"
                                   "{\"ok\": true, \"fusion_stats\": {\"total_turns\": 17, \"lead_high_turns\": 6, \"lead_medium_turns\": 2, \"sidekick_free_turns\": 9}}"),
                 text("Pass 3 audit verified against live prompts: Custom Fusion Mode cuts "
                      "weekly burn from 9.17%/h to 2.28%/h (4.03x runway, 19h 44m vs 4h 54m "
                      "raw), `:8015` local mixture verification and continuation context "
                      "inheritance are active, and `:41435` bridge has routed 17 turns across "
                      "Argon Lead and Free Sidekick.")]))

    # plain chat: no tool needed
    E.append(mk([user("what model are you"),
                 text("I am qwen3-coder:30b running locally on your RTX 4090 through "
                      "Ollama, behind the ailm chat shim on :8015.")]))

    return E


def main():
    assert MODEL_DIR, "pass the HF model dir for the tokenizer"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Thinking models (qwen3.6/qwen3.8): the serve-time renderer draws the
    # prompt two ways — thinking-on ends the prompt at `<think>\n` and the
    # model must close an empty think before answering; reasoning_effort=none
    # (what the shim sends during gaming/convergence) pre-closes the think in
    # the prompt and the model answers directly. Emit a row per mode so the
    # adapter covers whichever render the server produces.
    probe = [{"role": "user", "content": "x"}]
    think_off_prompt = None
    try:
        p_off = tok.apply_chat_template(
            probe, tools=TOOLS, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        p_on = tok.apply_chat_template(
            probe, tools=TOOLS, tokenize=False,
            add_generation_prompt=True)
        if p_off != p_on:
            think_off_prompt = True
    except Exception:
        think_off_prompt = None

    rows = []

    def emit(prompt_msgs, m):
        prompt = tok.apply_chat_template(
            prompt_msgs, tools=TOOLS, tokenize=False,
            add_generation_prompt=True)
        full = tok.apply_chat_template(
            prompt_msgs + [m], tools=TOOLS, tokenize=False,
            add_generation_prompt=False)
        assert full.startswith(prompt), \
            "template render is not prefix-stable"
        rows.append({"prompt": prompt, "completion": full[len(prompt):]})
        if think_off_prompt:
            p_off = tok.apply_chat_template(
                prompt_msgs, tools=TOOLS, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            f_off = tok.apply_chat_template(
                prompt_msgs + [m], tools=TOOLS, tokenize=False,
                add_generation_prompt=False, enable_thinking=False)
            if f_off.startswith(p_off):
                rows.append(
                    {"prompt": p_off, "completion": f_off[len(p_off):]})

    for ep in episodes():
        prefix = []
        for m in ep:
            if m["role"] == "assistant":
                emit([{"role": "system", "content": SYSTEM}] + prefix, m)
            prefix.append(m)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(rows), "rows ->", OUT,
          "(dual think/no-think renders)" if think_off_prompt else "")
    print("--- sample completion ---")
    print(repr(rows[0]["completion"][:400]))


if __name__ == "__main__":
    main()