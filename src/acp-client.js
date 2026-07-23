/**
 * ACP Client — JSON-RPC 2.0 over stdio to Kiro CLI
 *
 * Implements the Agent Client Protocol (ACP) to communicate with Kiro CLI.
 * Protocol: newline-delimited JSON-RPC 2.0 on stdin/stdout.
 *
 * This variant of the client is used by the Ollama emulation server. Unlike a
 * chat bot that keeps a single persistent conversation, an Ollama server is
 * stateless: the client (e.g. Strands) resends the full message history on
 * every request. To emulate that faithfully we create a fresh ACP session per
 * request, so Kiro never sees duplicated context.
 *
 * @see https://kiro.dev/docs/cli/acp/
 * @see https://agentclientprotocol.com
 */

const { spawn } = require("child_process");
const { EventEmitter } = require("events");

const KIRO_CLI = process.env.KIRO_CLI_PATH || "kiro-cli";
const WORKSPACE = process.env.KIRO_WORKSPACE || process.cwd();
const REQUEST_TIMEOUT = Number(process.env.KIRO_TIMEOUT || 120_000);

class AcpClient extends EventEmitter {
  constructor() {
    super();
    this.proc = null;
    this.agentInfo = null;
    this.nextId = 0;
    this.pending = new Map();
    this.buffer = "";
  }

  /**
   * Start the ACP connection to Kiro CLI.
   *
   * Lifecycle:
   *   1. Spawn kiro-cli acp process
   *   2. Send initialize request (exchange capabilities)
   *   3. Ready to create sessions and accept prompts
   *
   * Note: sessions are created per request via createSession(), not here.
   */
  async start() {
    // const args = ["acp", "--trust-all-tools"];
    const args = ["acp"];
    const agent = process.env.KIRO_AGENT;
    if (agent) args.push("--agent", agent);

    // Model that Kiro CLI actually runs. This is the real model behind the
    // emulator — the model_id an Ollama client sends is only a label and does
    // not affect this. Leave KIRO_MODEL unset to use Kiro's default.
    const model = process.env.KIRO_MODEL;
    if (model) args.push("--model", model);

    // Optional reasoning effort: low | medium | high | xhigh | max.
    const effort = process.env.KIRO_EFFORT;
    if (effort) args.push("--effort", effort);

    this.proc = spawn(KIRO_CLI, args, {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, KIRO_LOG_LEVEL: "error" },
      cwd: WORKSPACE,
    });

    this.proc.stdout.on("data", (chunk) => this._onData(chunk));
    this.proc.stderr.on("data", (chunk) => {
      const msg = chunk.toString().trim();
      if (msg) console.error("[kiro]", msg);
    });
    this.proc.on("exit", (code) => {
      console.error(`[acp] kiro-cli exited (code ${code})`);
      // Reject any in-flight requests so callers don't hang forever.
      for (const [id, { reject, timeout }] of this.pending) {
        clearTimeout(timeout);
        reject(new Error(`kiro-cli exited (code ${code})`));
        this.pending.delete(id);
      }
      this.emit("exit", code);
    });

    // Initialize — exchange capabilities
    const init = await this._request("initialize", {
      protocolVersion: 1,
      clientCapabilities: {
        fs: { readTextFile: true, writeTextFile: true },
        terminal: true,
      },
      clientInfo: { name: "kiro-acp-ollama-emulator", version: "1.0.0" },
    });
    this.agentInfo = init.agentInfo || null;
    console.log(
      `[acp] Connected: ${init.agentInfo?.name || "kiro"} ${init.agentInfo?.version || ""}`.trim()
    );

    return this.agentInfo;
  }

  /**
   * Create a new ACP session and return its id.
   * A fresh session gives each Ollama request an isolated, stateless context.
   */
  async createSession() {
    const session = await this._request("session/new", {
      cwd: WORKSPACE,
      mcpServers: [],
    });
    return session.sessionId;
  }

  /**
   * Send a prompt to a specific session and collect the streamed response.
   *
   * ACP streams responses via session/update notifications:
   *   - agent_message_chunk: text fragments
   *   - tool_call: tool invocations (handled by Kiro internally)
   *
   * @param {string} sessionId  Session to prompt (from createSession()).
   * @param {string} text       The prompt text.
   * @param {(chunk: string) => void} [onChunk]  Optional per-fragment callback
   *        for live streaming to the HTTP client.
   * @returns {Promise<string>} The full concatenated response text.
   */
  prompt(sessionId, text, onChunk) {
    return new Promise((resolve, reject) => {
      const chunks = [];

      const onNotification = (_method, params) => {
        if (params.sessionId && params.sessionId !== sessionId) return;
        const u = params.update;
        if (u?.sessionUpdate === "agent_message_chunk" && u.content?.text) {
          chunks.push(u.content.text);
          if (onChunk) {
            try {
              onChunk(u.content.text);
            } catch (_) {}
          }
        }
      };

      this.on("notification", onNotification);

      this._request("session/prompt", {
        sessionId,
        prompt: [{ type: "text", text }],
      })
        .then(() => {
          this.removeListener("notification", onNotification);
          resolve(chunks.join(""));
        })
        .catch((err) => {
          this.removeListener("notification", onNotification);
          reject(err);
        });
    });
  }

  /**
   * Convenience: create a fresh session, prompt it, and return the response.
   * This is the stateless entry point used per Ollama /api/chat request.
   */
  async chat(text, onChunk) {
    const sessionId = await this.createSession();
    return this.prompt(sessionId, text, onChunk);
  }

  stop() {
    if (this.proc) {
      this.proc.kill();
      this.proc = null;
    }
  }

  // --- JSON-RPC transport layer ---

  _request(method, params) {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Timeout: ${method}`));
      }, REQUEST_TIMEOUT);
      this.pending.set(id, { resolve, reject, timeout });
      this._send({ jsonrpc: "2.0", id, method, params });
    });
  }

  _send(msg) {
    if (!this.proc?.stdin.writable) throw new Error("ACP process not running");
    this.proc.stdin.write(JSON.stringify(msg) + "\n");
  }

  _onData(chunk) {
    this.buffer += chunk.toString();
    let idx;
    while ((idx = this.buffer.indexOf("\n")) !== -1) {
      const line = this.buffer.slice(0, idx).trim();
      this.buffer = this.buffer.slice(idx + 1);
      if (!line) continue;
      try {
        this._handleMessage(JSON.parse(line));
      } catch (_) {}
    }
  }

  _handleMessage(msg) {
    // Response to our request
    if (msg.id !== undefined && this.pending.has(msg.id)) {
      const { resolve, reject, timeout } = this.pending.get(msg.id);
      clearTimeout(timeout);
      this.pending.delete(msg.id);
      if (msg.error) reject(new Error(msg.error.message));
      else resolve(msg.result);
      return;
    }
    // Server request (needs response) — e.g., fs/readTextFile
    if (msg.id !== undefined && msg.method) {
      this._handleServerRequest(msg);
      return;
    }
    // Notification (no id)
    if (msg.method) {
      this.emit("notification", msg.method, msg.params || {});
    }
  }

  async _handleServerRequest(msg) {
    try {
      const result = await this._dispatch(msg.method, msg.params || {});
      this._send({ jsonrpc: "2.0", id: msg.id, result });
    } catch (err) {
      this._send({
        jsonrpc: "2.0",
        id: msg.id,
        error: { code: -32000, message: err.message },
      });
    }
  }

  /**
   * Handle requests FROM Kiro (tool execution).
   * Kiro may request file reads, writes, or terminal commands.
   */
  async _dispatch(method, params) {
    const fs = require("fs");
    const path = require("path");

    switch (method) {
      case "fs/readTextFile": {
        const p = params.path.startsWith("/")
          ? params.path
          : path.join(WORKSPACE, params.path);
        return { content: fs.readFileSync(p, "utf-8") };
      }
      case "fs/writeTextFile": {
        const p = params.path.startsWith("/")
          ? params.path
          : path.join(WORKSPACE, params.path);
        fs.mkdirSync(path.dirname(p), { recursive: true });
        fs.writeFileSync(p, params.content, "utf-8");
        return { success: true };
      }
      case "terminal/execute": {
        const { execSync } = require("child_process");
        const out = execSync(params.command, {
          cwd: WORKSPACE,
          timeout: 30_000,
          encoding: "utf-8",
        });
        return { output: out };
      }
      default:
        throw new Error(`Unsupported: ${method}`);
    }
  }
}

module.exports = AcpClient;
