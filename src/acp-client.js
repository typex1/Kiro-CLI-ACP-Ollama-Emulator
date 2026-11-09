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

// Kiro must act as a pure language model: every tool call is made by the
// client (e.g. Strands), never by Kiro itself. We therefore start kiro-cli with
// a tool-less agent shipped in this repo (.kiro/agents/ollama-emulator.json).
// Workspace agents are discovered relative to kiro-cli's working directory, so
// kiro-cli is spawned in the repo root (see start()).
const REPO_ROOT = require("path").resolve(__dirname, "..");
const DEFAULT_AGENT = "ollama-emulator";

/** JSON-RPC 2.0 "Method not found" — returned for requests we don't implement. */
class MethodNotFoundError extends Error {
  constructor(method) {
    super(`Method not found: ${method}`);
    this.code = -32601;
  }
}

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
    const args = ["acp"];
    // Tool-less agent by default; KIRO_AGENT overrides it (at your own risk —
    // an agent with tools lets Kiro act on the workspace by itself).
    const agent = process.env.KIRO_AGENT || DEFAULT_AGENT;
    args.push("--agent", agent);

    // Model that Kiro CLI actually runs. This is the real model behind the
    // emulator — the model_id an Ollama client sends is only a label and does
    // not affect this. Leave KIRO_MODEL unset to use Kiro's default.
    //
    // KIRO_MODEL is NOT passed as `--model`: that flag only applies to the
    // first ACP session, and this client opens a fresh session per request.
    // createSession() sets the model on every session instead.

    // Optional reasoning effort: low | medium | high | xhigh | max.
    // Note: like --model, --effort only applies to the first session, and it
    // makes kiro-cli use its other ACP engine (see _applyModel()).
    const effort = process.env.KIRO_EFFORT;
    if (effort) args.push("--effort", effort);

    this.proc = spawn(KIRO_CLI, args, {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, KIRO_LOG_LEVEL: "error" },
      // Repo root so kiro-cli discovers .kiro/agents/ollama-emulator.json.
      cwd: REPO_ROOT,
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

    // Initialize — exchange capabilities. We deliberately advertise NO file
    // system or terminal access: Kiro must not act on the workspace.
    const init = await this._request("initialize", {
      protocolVersion: 1,
      clientCapabilities: {
        fs: { readTextFile: false, writeTextFile: false },
        terminal: false,
      },
      clientInfo: { name: "kiro-acp-ollama-emulator", version: "1.0.0" },
    });
    this.agentInfo = init.agentInfo || null;
    console.log(
      `[acp] Connected: ${init.agentInfo?.name || "kiro"} ${init.agentInfo?.version || ""}`.trim()
    );
    console.log(`[acp] Agent: ${agent} (Kiro's own tool calls are rejected)`);
    if (process.env.KIRO_MODEL) {
      console.log(`[acp] Model: ${process.env.KIRO_MODEL} (applied to every session)`);
    }

    return this.agentInfo;
  }

  /**
   * Create a new ACP session and return its id.
   * A fresh session gives each Ollama request an isolated, stateless context.
   * If KIRO_MODEL is set, the model is applied to this session (see _applyModel).
   */
  async createSession() {
    const session = await this._request("session/new", {
      cwd: WORKSPACE,
      mcpServers: [],
    });
    const model = process.env.KIRO_MODEL;
    if (model) await this._applyModel(session, model);
    return session.sessionId;
  }

  /**
   * Set the model for one session. kiro-cli has two ACP engines that expose
   * this differently; pick the call from the shape of the session/new result:
   *
   *   - default engine: result has `configOptions` with an option
   *     {id: "model", category: "model"} → session/set_config_option
   *   - engine used when started with --effort ("Kiro CLI Agent 3.0.0"):
   *     result has `models` → session/set_model
   *
   * An unknown model id fails the request with the list of valid ids, rather
   * than silently running on the default model.
   */
  async _applyModel(session, model) {
    const { sessionId } = session;
    const option = (session.configOptions || []).find(
      (o) => o.category === "model" || o.id === "model"
    );

    if (option) {
      const valid = (option.options || []).map((o) => o.value);
      if (valid.length && !valid.includes(model)) {
        throw new Error(`KIRO_MODEL "${model}" is not available. Valid: ${valid.join(", ")}`);
      }
      if (option.currentValue === model) return;
      await this._request("session/set_config_option", {
        sessionId,
        configId: option.id,
        value: model,
      });
      return;
    }

    if (session.models) {
      const valid = (session.models.availableModels || []).map((m) => m.modelId);
      if (valid.length && !valid.includes(model)) {
        throw new Error(`KIRO_MODEL "${model}" is not available. Valid: ${valid.join(", ")}`);
      }
      if (session.models.currentModelId === model) return;
      await this._request("session/set_model", { sessionId, modelId: model });
      return;
    }

    if (!this._warnedNoModelSupport) {
      console.error(
        `[acp] KIRO_MODEL="${model}" ignored: this kiro-cli does not report a selectable model`
      );
      this._warnedNoModelSupport = true;
    }
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
    // JSON-RPC 2.0: a message WITH "method" is a request (has id) or a
    // notification (no id) from Kiro; only messages WITHOUT "method" are
    // responses to our requests. Check "method" first — Kiro numbers its own
    // requests independently, so their ids can collide with our pending ids
    // (e.g. Kiro's _kiro/terminal/shell_type arriving as id 1 while our
    // session/new is pending as id 1 was mistaken for its response).

    // Server request (needs response) — e.g., fs/readTextFile
    if (msg.method && msg.id !== undefined) {
      this._handleServerRequest(msg);
      return;
    }
    // Notification (no id)
    if (msg.method) {
      const u = msg.params?.update;
      if (u?.sessionUpdate === "tool_call") {
        // Should never happen with the tool-less agent; make it visible.
        console.error(
          `[acp] WARNING: Kiro attempted its own tool call: ${u.title || u.kind || "?"}`
        );
      }
      this.emit("notification", msg.method, msg.params || {});
      return;
    }
    // Response to our request
    if (msg.id !== undefined && this.pending.has(msg.id)) {
      const { resolve, reject, timeout } = this.pending.get(msg.id);
      clearTimeout(timeout);
      this.pending.delete(msg.id);
      if (msg.error) reject(new Error(msg.error.message));
      else resolve(msg.result);
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
        // -32601 for unknown methods (see MethodNotFoundError), -32000 otherwise.
        error: { code: err.code ?? -32000, message: err.message },
      });
    }
  }

  /**
   * Handle requests FROM Kiro.
   *
   * Kiro must not use its own tools — tool calls belong to the client (e.g.
   * Strands). So file system and terminal requests are refused, and every
   * permission request is answered with "reject". This is defense in depth on
   * top of the tool-less agent (.kiro/agents/ollama-emulator.json).
   */
  async _dispatch(method, params) {
    const path = require("path");

    switch (method) {
      case "session/request_permission": {
        // Pick the agent-offered reject option (prefer a one-off reject).
        const options = params.options || [];
        const reject =
          options.find((o) => o.kind === "reject_once") ||
          options.find((o) => o.kind === "reject_always");
        console.error(
          `[acp] Rejected Kiro tool permission: ${params.toolCall?.title || "?"}`
        );
        return reject
          ? { outcome: { outcome: "selected", optionId: reject.optionId } }
          : { outcome: { outcome: "cancelled" } };
      }
      case "fs/read_text_file":
      case "fs/readTextFile":
      case "fs/write_text_file":
      case "fs/writeTextFile":
      case "terminal/create":
      case "terminal/execute": {
        console.error(`[acp] Refused Kiro tool request: ${method}`);
        const err = new Error(
          `${method} is disabled: this bridge does not let Kiro use its own tools`
        );
        err.code = -32000;
        throw err;
      }
      // Kiro-specific extension: Kiro asks which shell its terminal commands
      // will run in, so it can phrase them correctly. Answer with the basename
      // of $SHELL (e.g. "bash", "zsh"), falling back to "bash".
      case "_kiro/terminal/shell_type": {
        const shellType = path.basename(process.env.SHELL || "bash");
        return { shellType };
      }
      default:
        // ACP reserves "_"-prefixed methods for vendor extensions. A client
        // that doesn't implement one should answer JSON-RPC -32601 (Method
        // not found) rather than a generic failure, so the agent can carry on.
        // Log it so the next missing extension is easy to spot.
        console.error(
          `[acp] unsupported request from Kiro: ${method} params=${JSON.stringify(params)}`
        );
        throw new MethodNotFoundError(method);
    }
  }
}

module.exports = AcpClient;
