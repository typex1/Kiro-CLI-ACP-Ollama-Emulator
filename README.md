# Kiro CLI ACP Ollama Emulator

Expose [Kiro CLI](https://kiro.dev/docs/cli/) as a **local Ollama-compatible server**, using the **Agent Client Protocol (ACP)** under the hood.

Point any tool that speaks the Ollama REST API — such as the [Strands Agents](https://strandsagents.com/) `OllamaModel` provider — at `http://localhost:11435`, and it will talk to Kiro instead of a real local model.

## Why

If you're learning how Strands Agents work but don't have a local LLM or an API key for Bedrock / Anthropic / OpenAI, this server lets you build your first agent prototypes using only Kiro CLI as the backend. Your Strands code stays exactly as it would be for a real Ollama setup — you just change the port.

It runs on **port 11435** (the real Ollama uses 11434), so the two can coexist.

```
Strands Agent (OllamaModel) ──HTTP /api/chat──▶ this server ──ACP (JSON-RPC/stdio)──▶ kiro-cli acp ──▶ response
```

This project is adapted from a Telegram-to-Kiro bridge — same ACP client, different front end. Instead of Telegram messages, the client is now an Ollama HTTP API.

## Prerequisites

- Node.js 18+
- [Kiro CLI](https://kiro.dev/docs/cli/) installed and authenticated
- (For the example) Python with `strands-agents` installed

**No Pro license required** — ACP mode works with the free tier of Kiro CLI.

## Setup

Authenticate Kiro CLI once on the machine where the server runs:

```bash
kiro-cli chat
```

This opens a browser login flow. Once authenticated, the server can use `kiro-cli acp` without further interaction.

Then start the server:

```bash
cp .env_example .env   # optional — defaults work out of the box
npm start
```

You should see:

```
✅ Kiro ACP connected
🚀 Ready. Set your Ollama client host to http://127.0.0.1:11435
[ollama] Emulation server listening on http://127.0.0.1:11435
```

## Use it from Strands Agents

```python
from strands import Agent
from strands.models.ollama import OllamaModel

ollama_model = OllamaModel(
    host="http://localhost:11435",   # the bridge, not real Ollama (11434)
    model_id="kiro-acp",             # any label — the bridge ignores it
)

agent = Agent(model=ollama_model)
agent("Explain the agent loop in one paragraph.")
```

Runnable examples live in [`examples/`](examples/), numbered from simplest to most involved:

```bash
pip install -r examples/requirements.txt

python examples/1-strands-ollama-model.py   # plain chat, no tools
python examples/2-agent_with_tools.py        # built-in + custom tools
python examples/3-strands-mcp.py             # tools served over MCP
```

See [`examples/requirements.txt`](examples/requirements.txt) for the exact dependencies (example 3 also needs `uvx` and AWS credentials).

## Quick test with curl

```bash
# List the advertised model
curl http://localhost:11435/api/tags

# Non-streaming chat
curl http://localhost:11435/api/chat -d '{
  "model": "kiro-acp",
  "stream": false,
  "messages": [{"role": "user", "content": "Reply with exactly: PONG"}]
}'
```

## How it works

The server speaks a subset of the [Ollama REST API](https://github.com/ollama/ollama/blob/main/docs/api.md). For each `/api/chat` request it:

1. Flattens the incoming `messages` array (system + conversation history) into a single prompt.
2. Opens a **fresh ACP session** with Kiro and sends the prompt. A new session per request keeps the bridge stateless, matching how Ollama works (the client resends full history each call), so context is never duplicated.
3. Re-shapes Kiro's answer into the exact response schema the Ollama client library expects — streaming NDJSON chunks with a final `done` chunk carrying `prompt_eval_count`, `eval_count`, and `total_duration`.

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Health check (`Ollama is running`) |
| `GET` | `/api/version` | Advertised version |
| `GET` | `/api/tags` | List the bridged model |
| `GET` | `/api/ps` | List "running" models |
| `POST` | `/api/show` | Model details |
| `POST` | `/api/chat` | Chat completion (streaming or single JSON) |
| `POST` | `/api/generate` | Prompt completion (streaming or single JSON) |

### ACP methods used

| Method | Purpose |
|--------|---------|
| `initialize` | Exchange capabilities on startup |
| `session/new` | Create a fresh session per request |
| `session/prompt` | Send the flattened prompt to Kiro |

The bridge also handles server-initiated requests from Kiro (`fs/readTextFile`, `fs/writeTextFile`, `terminal/execute`) so Kiro can use its tools in the workspace.

## Configuration

All optional — defaults work out of the box. Set via `.env` or environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_PORT` | `11435` | Port for the emulation server |
| `OLLAMA_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` to expose on your network) |
| `OLLAMA_MODEL_NAME` | `kiro-acp` | Model name advertised in `/api/tags` |
| `OLLAMA_TOOLCALLS` | `on` | Tool-call emulation; set `off` to ignore request `tools` |
| `KIRO_CLI_PATH` | `kiro-cli` | Path to the kiro-cli binary |
| `KIRO_WORKSPACE` | cwd | Working directory for Kiro |
| `KIRO_AGENT` | — | Named Kiro agent to use for sessions |
| `KIRO_MODEL` | Kiro default | Model Kiro CLI actually runs, set on every ACP session. **This picks the real model** — see below |
| `KIRO_EFFORT` | — | Reasoning effort (`--effort`): `low` / `medium` / `high` / `xhigh` / `max`. Only applies to the first ACP session — see below |
| `KIRO_TIMEOUT` | `120000` | ACP request timeout (ms) |

### Which model actually answers?

The `model_id` your Ollama client sends (e.g. `OllamaModel(model_id="...")`) is **only a label** — the bridge ignores it. The real model is chosen by Kiro CLI. To control it, set `KIRO_MODEL`:

```bash
KIRO_MODEL=claude-haiku-4.5 npm start
```

List the valid IDs for your account with `kiro-cli chat --list-models`. An unknown ID fails each request with the list of valid IDs instead of silently using the default. Leave `KIRO_MODEL` unset to use Kiro's default (`auto`).

How it works: the bridge opens a fresh ACP session per request, and `kiro-cli acp --model` only applies to the first session. So instead of passing `--model`, the bridge sets the model right after each `session/new`: via `session/set_config_option` (`configId: "model"`) on kiro-cli's default ACP engine, or via `session/set_model` on the engine kiro-cli switches to when started with `--effort`. The same first-session-only limit applies to `KIRO_EFFORT`, and there is no per-session effort option to work around it.

## Structured output

Strands' `agent.structured_output(...)` sends a JSON schema in the request's `format` field with streaming disabled. The bridge honors this: it asks Kiro to return JSON-only output conforming to the schema and responds with a single non-streamed JSON object.

## Tool calling (function calling)

The bridge supports Strands tool calling through prompt-based emulation. When a
request includes a `tools` list, the server:

1. Injects the tool catalog into the prompt and asks Kiro to act as a
   function-calling model — either emit a strict JSON directive
   (`{"tool_calls":[{"name":..., "arguments":{...}}]}`) or answer in plain text.
2. Buffers Kiro's full reply and parses out that directive (tolerant of code
   fences and surrounding prose).
3. Re-shapes it into Ollama `tool_calls`, which Strands executes locally. The
   tool result comes back on the next request and Kiro produces the final answer.

Try it with [`examples/2-agent_with_tools.py`](examples/2-agent_with_tools.py), which mixes
built-in tools (`calculator`, `current_time`) with a custom `letter_counter`:

```bash
python examples/2-agent_with_tools.py
# The agent calls each tool and reports the time, the division result,
# and the count of "r" in "strawberry".
```

MCP-backed tools work the same way — see [`examples/3-strands-mcp.py`](examples/3-strands-mcp.py).
Strands discovers the MCP tools and passes them as ordinary function specs, so the
bridge handles them with no changes.

Set `OLLAMA_TOOLCALLS=off` to disable this and ignore tools entirely.

**Caveats — this is best-effort, not native tool calling:**

- Reliability depends on Kiro following the "emit JSON, don't use your own
  tools" instruction. Clear, well-described tools work best; it can occasionally
  answer directly instead of calling a tool, or vice versa.
- Tool-mode requests are **not streamed token-by-token** — the reply is buffered
  so it can be parsed before responding.
- Parallel/nested tool calls and streaming partial arguments are not guaranteed.

For production tool-driven agents, a model provider with native function calling
is still the better choice. This mode is aimed at learning and prototyping.

## Limitations

- **Images are not forwarded** — image message parts are noted as omitted in the prompt.
- **Token counts are estimates** (~4 characters per token); Kiro's ACP responses don't expose exact token usage.
- Responses come from Kiro CLI, so latency and behavior reflect Kiro, not a small local model.

## Run the tests

Runs without kiro-cli (uses a fake ACP client) to verify prompt flattening and the HTTP/streaming layer:

```bash
npm test
```

## Deploying as a systemd service

```bash
sudo cp kiro-ollama-emulator.service /etc/systemd/system/
sudo systemctl enable --now kiro-ollama-emulator
```

## Security note

By default the server binds to `127.0.0.1` (localhost only) and has **no authentication**. If you set `OLLAMA_HOST=0.0.0.0` to expose it on your network, anyone who can reach the port can drive Kiro CLI in your workspace — including file writes and terminal commands via Kiro's tools. Only do this on trusted networks, and consider a firewall or reverse proxy with auth in front.
