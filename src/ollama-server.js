/**
 * Ollama Emulation Server — HTTP front end that speaks the Ollama REST API
 * but answers with Kiro CLI (via ACP) instead of a local model.
 *
 * The goal: let tools that expect a local Ollama server (such as the Strands
 * Agents `OllamaModel` provider) talk to Kiro CLI with zero code changes —
 * just point them at http://localhost:11435.
 *
 * Endpoints implemented (subset of the Ollama API):
 *   GET  /               → health check ("Ollama is running")
 *   GET  /api/version    → advertised version
 *   GET  /api/tags       → list "installed" models
 *   POST /api/show       → model details
 *   POST /api/chat       → chat completion (streaming NDJSON or single JSON)
 *   POST /api/generate   → prompt completion (streaming NDJSON or single JSON)
 *
 * The chat/generate handlers flatten the incoming message history into a
 * single prompt, forward it to Kiro over ACP, and re-shape Kiro's answer into
 * the exact response schema the Ollama client library parses.
 *
 * @see https://github.com/ollama/ollama/blob/main/docs/api.md
 */

const http = require("http");

const VERSION = "0.0.0-kiro-acp";

class OllamaServer {
  /**
   * @param {import('./acp-client')} acpClient  Connected ACP client.
   * @param {object} [opts]
   * @param {number} [opts.port]      Listen port (default 11435).
   * @param {string} [opts.host]      Bind address (default 127.0.0.1).
   * @param {string} [opts.modelName] Model name advertised in /api/tags.
   */
  constructor(acpClient, opts = {}) {
    this.acp = acpClient;
    // Note: 0 is a valid port (OS-assigned ephemeral), so don't use `||`.
    this.port = opts.port != null ? opts.port : 11435;
    this.host = opts.host || "127.0.0.1";
    this.modelName = opts.modelName || "kiro-acp";
    // Translate client-supplied tools into a function-calling protocol Kiro can
    // follow via prompt engineering. On by default; disable to ignore tools.
    this.toolCalls = opts.toolCalls !== false;
    this.server = null;
    // Serialize prompts: one ACP turn at a time keeps sessions coherent.
    this._queue = Promise.resolve();
  }

  start() {
    this.server = http.createServer((req, res) => this._route(req, res));
    this.server.listen(this.port, this.host, () => {
      console.log(
        `[ollama] Emulation server listening on http://${this.host}:${this.port}`
      );
      console.log(`[ollama] Advertised model: "${this.modelName}"`);
    });
    this.server.on("error", (err) => {
      if (err.code === "EADDRINUSE") {
        console.error(
          `[ollama] Port ${this.port} is already in use. ` +
            `Is a real Ollama server (or another instance) running? ` +
            `Set OLLAMA_PORT to a free port.`
        );
        process.exit(1);
      }
      console.error("[ollama] Server error:", err.message);
    });
  }

  stop() {
    if (this.server) this.server.close();
  }

  // --- Routing ---

  async _route(req, res) {
    const url = (req.url || "/").split("?")[0];
    const method = req.method || "GET";

    try {
      if (method === "GET" && (url === "/" || url === "")) {
        return this._text(res, 200, "Ollama is running");
      }
      if (method === "HEAD" && url === "/") {
        res.writeHead(200);
        return res.end();
      }
      if (method === "GET" && url === "/api/version") {
        return this._json(res, 200, { version: VERSION });
      }
      if (method === "GET" && url === "/api/tags") {
        return this._json(res, 200, { models: [this._modelEntry()] });
      }
      if (method === "GET" && url === "/api/ps") {
        return this._json(res, 200, { models: [this._modelEntry(true)] });
      }
      if (method === "POST" && url === "/api/show") {
        await this._readBody(req); // drain
        return this._json(res, 200, this._modelDetails());
      }
      if (method === "POST" && url === "/api/chat") {
        const body = await this._readBody(req);
        return this._handleChat(res, body);
      }
      if (method === "POST" && url === "/api/generate") {
        const body = await this._readBody(req);
        return this._handleGenerate(res, body);
      }

      return this._json(res, 404, { error: `unknown endpoint: ${method} ${url}` });
    } catch (err) {
      console.error("[ollama] Request error:", err.message);
      if (!res.headersSent) {
        this._json(res, 500, { error: err.message });
      } else {
        try {
          res.end();
        } catch (_) {}
      }
    }
  }

  // --- /api/chat ---

  async _handleChat(res, body) {
    const model = body.model || this.modelName;
    const stream = body.stream !== false; // Ollama default is true
    const structured = body.format && typeof body.format === "object";
    const tools = Array.isArray(body.tools) ? body.tools : [];
    const toolMode = this.toolCalls && tools.length > 0 && !structured;

    let prompt;
    try {
      prompt = buildPromptFromMessages(body.messages || [], structured ? body.format : null);
    } catch (err) {
      return this._json(res, 400, { error: err.message });
    }

    // Tool-calling mode: ask Kiro to act as a function-calling model, then
    // parse its reply back into Ollama tool_calls. Requires the full reply
    // before we can parse, so this path always buffers (no live streaming).
    if (toolMode) {
      prompt = withToolInstruction(prompt, tools);
      return this._runToolMode(res, model, prompt, stream);
    }

    // Strands' structured_output sends stream:false + a JSON schema in `format`.
    // Honor that by returning a single non-streamed JSON object.
    const wantStream = stream && !structured;

    await this._run(res, model, prompt, wantStream, (content) => ({
      model,
      created_at: new Date().toISOString(),
      message: { role: "assistant", content },
      done: false,
    }));
  }

  /** Queue and run a tool-mode chat turn (buffers the full reply, then parses). */
  async _runToolMode(res, model, prompt, stream) {
    const task = this._queue.then(() =>
      this._executeToolMode(res, model, prompt, stream)
    );
    this._queue = task.catch(() => {});
    return task;
  }

  async _executeToolMode(res, model, prompt, stream) {
    const started = Date.now();
    const promptTokens = estimateTokens(prompt);

    let full;
    try {
      full = await this.acp.chat(prompt); // buffer entire reply
    } catch (err) {
      if (stream) {
        res.writeHead(200, { "Content-Type": "application/x-ndjson" });
        res.write(JSON.stringify({ error: acpError(err) }) + "\n");
        return res.end();
      }
      return this._json(res, 500, { error: acpError(err) });
    }

    const toolCalls = parseToolCalls(full);
    const created_at = new Date().toISOString();

    // When Kiro asked for a tool, the assistant message carries tool_calls and
    // no user-facing text. Otherwise it's a normal text answer.
    const message = toolCalls.length
      ? { role: "assistant", content: "", tool_calls: toolCalls }
      : { role: "assistant", content: full };

    const finalExtra = this._finalFields(started, promptTokens, full);
    if (toolCalls.length) finalExtra.done_reason = "tool_calls";

    if (!stream) {
      return this._json(res, 200, { model, created_at, message, ...finalExtra });
    }

    res.writeHead(200, {
      "Content-Type": "application/x-ndjson",
      "Transfer-Encoding": "chunked",
    });
    // First line: the content (tool_calls or text). Strands reads tool_calls
    // off any streamed event, so a single carrier chunk is sufficient.
    res.write(JSON.stringify({ model, created_at, message, done: false }) + "\n");
    // Final line: closing metadata (empty message, no tool_calls).
    res.write(
      JSON.stringify({
        model,
        created_at: new Date().toISOString(),
        message: { role: "assistant", content: "" },
        ...finalExtra,
      }) + "\n"
    );
    return res.end();
  }

  // --- /api/generate ---

  async _handleGenerate(res, body) {
    const model = body.model || this.modelName;
    const stream = body.stream !== false;
    const structured = body.format && typeof body.format === "object";

    let prompt = typeof body.prompt === "string" ? body.prompt : "";
    if (body.system) prompt = `${body.system}\n\n${prompt}`;
    if (structured) prompt = withJsonInstruction(prompt, body.format);

    const wantStream = stream && !structured;

    await this._run(res, model, prompt, wantStream, (content) => ({
      model,
      created_at: new Date().toISOString(),
      response: content,
      done: false,
    }));
  }

  /**
   * Shared execution path for chat/generate.
   *
   * @param res
   * @param {string} model
   * @param {string} prompt
   * @param {boolean} wantStream
   * @param {(content: string) => object} shapeChunk  Builds the per-chunk body
   *        (chat uses `message`, generate uses `response`).
   */
  async _run(res, model, prompt, wantStream, shapeChunk) {
    const started = Date.now();
    const promptTokens = estimateTokens(prompt);

    // Queue prompts so concurrent clients don't interleave ACP turns.
    const task = this._queue.then(() =>
      this._execute(res, model, prompt, wantStream, shapeChunk, started, promptTokens)
    );
    // Keep the chain alive even if this task rejects.
    this._queue = task.catch(() => {});
    return task;
  }

  async _execute(res, model, prompt, wantStream, shapeChunk, started, promptTokens) {
    if (wantStream) {
      res.writeHead(200, {
        "Content-Type": "application/x-ndjson",
        "Transfer-Encoding": "chunked",
      });

      let full = "";
      try {
        await this.acp.chat(prompt, (chunk) => {
          full += chunk;
          res.write(JSON.stringify(shapeChunk(chunk)) + "\n");
        });
      } catch (err) {
        // Mid-stream failure: emit an error line the ollama client will raise on.
        res.write(JSON.stringify({ error: acpError(err) }) + "\n");
        return res.end();
      }

      const finalChunk = shapeChunk("");
      Object.assign(finalChunk, this._finalFields(started, promptTokens, full));
      res.write(JSON.stringify(finalChunk) + "\n");
      return res.end();
    }

    // Non-streaming: collect everything, return one JSON object.
    let full;
    try {
      full = await this.acp.chat(prompt);
    } catch (err) {
      return this._json(res, 500, { error: acpError(err) });
    }
    const finalChunk = shapeChunk(full);
    Object.assign(finalChunk, this._finalFields(started, promptTokens, full));
    return this._json(res, 200, finalChunk);
  }

  /**
   * Fields required on the final chunk. Strands reads prompt_eval_count,
   * eval_count and total_duration off the last streamed event to build usage
   * metrics, so these must be present integers.
   */
  _finalFields(started, promptTokens, full) {
    const elapsedMs = Date.now() - started;
    const evalTokens = estimateTokens(full);
    return {
      done: true,
      done_reason: "stop",
      total_duration: elapsedMs * 1e6, // ns
      load_duration: 0,
      prompt_eval_count: promptTokens,
      prompt_eval_duration: 0,
      eval_count: evalTokens,
      eval_duration: elapsedMs * 1e6, // ns
    };
  }

  // --- Model metadata helpers ---

  _modelEntry(running = false) {
    const base = {
      name: `${this.modelName}:latest`,
      model: `${this.modelName}:latest`,
      modified_at: new Date().toISOString(),
      size: 0,
      digest: "kiro-acp-bridge",
      details: {
        parent_model: "",
        format: "acp",
        family: "kiro",
        families: ["kiro"],
        parameter_size: "N/A",
        quantization_level: "N/A",
      },
    };
    if (running) {
      base.expires_at = new Date(Date.now() + 5 * 60_000).toISOString();
      base.size_vram = 0;
    }
    return base;
  }

  _modelDetails() {
    return {
      license: "Bridged via Kiro CLI (ACP).",
      modelfile: `# Kiro ACP bridge for "${this.modelName}"`,
      parameters: "",
      template: "{{ .Prompt }}",
      details: this._modelEntry().details,
      model_info: {},
      capabilities: ["completion"],
    };
  }

  // --- Low-level response helpers ---

  _readBody(req) {
    return new Promise((resolve, reject) => {
      let data = "";
      req.on("data", (c) => {
        data += c;
        if (data.length > 50 * 1024 * 1024) {
          reject(new Error("request body too large"));
          req.destroy();
        }
      });
      req.on("end", () => {
        if (!data) return resolve({});
        try {
          resolve(JSON.parse(data));
        } catch (err) {
          reject(new Error("invalid JSON body"));
        }
      });
      req.on("error", reject);
    });
  }

  _json(res, status, obj) {
    const payload = JSON.stringify(obj);
    res.writeHead(status, {
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(payload),
    });
    res.end(payload);
  }

  _text(res, status, str) {
    res.writeHead(status, { "Content-Type": "text/plain" });
    res.end(str);
  }
}

// --- Prompt shaping ---

/**
 * Flatten an Ollama `messages` array into a single prompt string for Kiro.
 *
 * Ollama (and Strands) are stateless: the full history is resent every call.
 * We render it as a readable transcript so Kiro has the conversation context
 * within one fresh ACP session.
 *
 * @param {Array<{role:string, content?:string, images?:any[]}>} messages
 * @param {*} format  Optional response format (json / JSON schema).
 * @returns {string}
 */
function buildPromptFromMessages(messages, format) {
  const systemParts = [];
  const turns = [];

  for (const msg of messages) {
    const role = msg.role || "user";
    let content = typeof msg.content === "string" ? msg.content : "";

    if (Array.isArray(msg.images) && msg.images.length) {
      content += `${content ? "\n" : ""}[image attachment omitted — not supported by the Kiro ACP bridge]`;
    }
    // A prior assistant turn that called tools (function-calling loop).
    if (Array.isArray(msg.tool_calls) && msg.tool_calls.length) {
      const calls = msg.tool_calls
        .map((tc) => {
          const fn = tc.function || {};
          const argsStr = JSON.stringify(fn.arguments || {});
          return `${fn.name}(${argsStr})`;
        })
        .join(", ");
      const line = `You called function(s): ${calls}`;
      content = content ? `${content}\n${line}` : line;
    }

    if (!content) continue;

    if (role === "system") {
      systemParts.push(content);
    } else if (role === "tool") {
      // Result returned by the client after executing a requested tool call.
      turns.push(`Function result: ${content}`);
    } else if (role === "assistant") {
      turns.push(`Assistant: ${content}`);
    } else {
      turns.push(`User: ${content}`);
    }
  }

  let prompt;
  // Common simple case: a single user turn (with optional system prompt).
  const userTurns = turns.filter((t) => t.startsWith("User:"));
  if (turns.length === 1 && userTurns.length === 1) {
    prompt = turns[0].replace(/^User:\s*/, "");
  } else {
    prompt = turns.join("\n\n");
  }

  if (systemParts.length) {
    prompt = `${systemParts.join("\n\n")}\n\n${prompt}`;
  }

  if (format && typeof format === "object") {
    prompt = withJsonInstruction(prompt, format);
  } else if (format === "json") {
    prompt = withJsonInstruction(prompt, null);
  }

  return prompt.trim();
}

/**
 * Append an instruction asking for JSON-only output. Used for
 * `format: "json"` and Strands' structured_output (which passes a JSON schema).
 */
function withJsonInstruction(prompt, schema) {
  let instruction =
    "\n\nRespond with ONLY a single valid JSON value and nothing else. " +
    "Do not wrap it in markdown code fences or add any explanatory text.";
  if (schema) {
    instruction += ` The JSON must conform to this JSON Schema:\n${JSON.stringify(
      schema
    )}`;
  }
  return prompt + instruction;
}

/**
 * Append a function-calling protocol instruction so Kiro behaves like a
 * tool-using model. `tools` is the Ollama request's tools array:
 *   [{ type: "function", function: { name, description, parameters } }]
 *
 * We ask Kiro to either emit a strict JSON tool-call directive or a plain-text
 * final answer. The bridge parses the directive back into Ollama `tool_calls`.
 */
function withToolInstruction(prompt, tools) {
  const catalog = tools
    .map((t) => {
      const fn = (t && t.function) || {};
      return {
        name: fn.name,
        description: fn.description || "",
        parameters: fn.parameters || {},
      };
    })
    .filter((f) => f.name);

  const instruction = [
    "",
    "",
    "You have access to the following functions (tools). Use them to fulfill the",
    "user's request instead of answering from your own knowledge and instead of",
    "using any of your own built-in tools (do not read files or run commands yourself):",
    "",
    JSON.stringify(catalog, null, 2),
    "",
    "Decide your next step:",
    "- If you need to call one or more functions, respond with ONLY this JSON object",
    "  and nothing else (no prose, no markdown code fences):",
    '  {"tool_calls": [{"name": "<function_name>", "arguments": { <args matching the schema> }}]}',
    "- If a 'Function result:' for what you need is already present above and you can",
    "  now answer, respond with the final answer as plain text (no JSON).",
    "",
    "Only call functions that are listed above. Output the JSON on a single logical",
    "object with no surrounding text when calling a function.",
  ].join("\n");

  return prompt + instruction;
}

/**
 * Parse Kiro's reply for a tool-call directive and normalize it to Ollama's
 * shape: [{ function: { name, arguments } }] where arguments is an object.
 * Tolerant of code fences and surrounding prose. Returns [] if none found.
 */
function parseToolCalls(text) {
  if (!text) return [];

  const normalize = (call) => {
    if (!call || typeof call !== "object") return null;
    // Accept both {name, arguments} and Ollama-native {function:{name,arguments}}.
    const fn = call.function && typeof call.function === "object" ? call.function : call;
    const name = fn.name;
    if (!name || typeof name !== "string") return null;
    let args = fn.arguments !== undefined ? fn.arguments : fn.parameters;
    if (typeof args === "string") {
      try {
        args = JSON.parse(args);
      } catch (_) {
        /* keep string; wrap below */
      }
    }
    if (args === undefined || args === null) args = {};
    if (typeof args !== "object" || Array.isArray(args)) args = { value: args };
    return { function: { name, arguments: args } };
  };

  for (const obj of extractJsonObjects(text)) {
    if (Array.isArray(obj.tool_calls)) {
      const calls = obj.tool_calls.map(normalize).filter(Boolean);
      if (calls.length) return calls;
    }
    if (Array.isArray(obj.tool_call)) {
      const calls = obj.tool_call.map(normalize).filter(Boolean);
      if (calls.length) return calls;
    }
    // A bare single call object: {name, arguments} or {function:{...}}.
    if (obj.name || (obj.function && obj.function.name)) {
      const call = normalize(obj);
      if (call) return [call];
    }
  }
  return [];
}

/**
 * Extract candidate JSON objects from arbitrary text via brace matching.
 * Strips ```json fences first, then scans for balanced top-level {...} blocks.
 */
function extractJsonObjects(text) {
  const cleaned = text.replace(/```(?:json)?/gi, "");
  const objects = [];
  let depth = 0;
  let start = -1;
  let inStr = false;
  let escape = false;

  for (let i = 0; i < cleaned.length; i++) {
    const ch = cleaned[i];
    if (inStr) {
      if (escape) escape = false;
      else if (ch === "\\") escape = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') {
      inStr = true;
    } else if (ch === "{") {
      if (depth === 0) start = i;
      depth++;
    } else if (ch === "}") {
      if (depth > 0) {
        depth--;
        if (depth === 0 && start !== -1) {
          const slice = cleaned.slice(start, i + 1);
          try {
            objects.push(JSON.parse(slice));
          } catch (_) {
            /* skip malformed block */
          }
          start = -1;
        }
      }
    }
  }
  return objects;
}

/** Rough token estimate (~4 chars/token), floored at 1 to satisfy consumers. */
function estimateTokens(str) {
  if (!str) return 1;
  return Math.max(1, Math.ceil(str.length / 4));
}

/** Normalize an ACP error into a message string for the client. */
function acpError(err) {
  return err && err.message ? err.message : String(err);
}

module.exports = OllamaServer;
module.exports.buildPromptFromMessages = buildPromptFromMessages;
module.exports.estimateTokens = estimateTokens;
module.exports.parseToolCalls = parseToolCalls;
module.exports.withToolInstruction = withToolInstruction;
