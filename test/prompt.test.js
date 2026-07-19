/**
 * Lightweight tests for the Ollama emulation layer.
 *
 * Runs without kiro-cli by injecting a fake ACP client. Verifies:
 *   - prompt flattening from Ollama message arrays
 *   - /api/tags shape
 *   - /api/chat streaming NDJSON (chunks + final metadata fields)
 *   - /api/chat non-streaming single JSON
 *
 * Run: npm test   (or: node test/prompt.test.js)
 */

const assert = require("assert");
const http = require("http");
const OllamaServer = require("../src/ollama-server");
const {
  buildPromptFromMessages,
  estimateTokens,
  parseToolCalls,
  withToolInstruction,
} = require("../src/ollama-server");

let passed = 0;
function ok(name) {
  passed++;
  console.log(`  ✓ ${name}`);
}

// --- Fake ACP client: echoes a fixed reply, streaming word by word. ---
class FakeAcp {
  constructor(reply = "Hello from Kiro.") {
    this.reply = reply;
    this.lastPrompt = null;
  }
  async chat(prompt, onChunk) {
    this.lastPrompt = prompt;
    const words = this.reply.split(" ");
    let full = "";
    for (let i = 0; i < words.length; i++) {
      const frag = (i === 0 ? "" : " ") + words[i];
      full += frag;
      if (onChunk) onChunk(frag);
    }
    return full;
  }
}

function request(port, path, method, body) {
  return new Promise((resolve, reject) => {
    const data = body ? JSON.stringify(body) : null;
    const req = http.request(
      { host: "127.0.0.1", port, path, method, headers: { "Content-Type": "application/json" } },
      (res) => {
        let buf = "";
        res.on("data", (c) => (buf += c));
        res.on("end", () => resolve({ status: res.statusCode, body: buf }));
      }
    );
    req.on("error", reject);
    if (data) req.write(data);
    req.end();
  });
}

// --- Unit: prompt flattening ---
function testPromptBuilding() {
  console.log("buildPromptFromMessages:");

  const single = buildPromptFromMessages([{ role: "user", content: "Hi there" }]);
  assert.strictEqual(single, "Hi there");
  ok("single user message → bare prompt");

  const withSystem = buildPromptFromMessages([
    { role: "system", content: "You are a pirate." },
    { role: "user", content: "Hello" },
  ]);
  assert.ok(withSystem.includes("You are a pirate."));
  assert.ok(withSystem.includes("Hello"));
  ok("system prompt is prepended");

  const multi = buildPromptFromMessages([
    { role: "user", content: "What is 2+2?" },
    { role: "assistant", content: "4" },
    { role: "user", content: "And times 3?" },
  ]);
  assert.ok(multi.includes("User: What is 2+2?"));
  assert.ok(multi.includes("Assistant: 4"));
  assert.ok(multi.includes("User: And times 3?"));
  ok("multi-turn history rendered as transcript");

  const jsonFmt = buildPromptFromMessages([{ role: "user", content: "give me json" }], {
    type: "object",
    properties: { name: { type: "string" } },
  });
  assert.ok(/JSON Schema/i.test(jsonFmt));
  ok("JSON schema format appends instruction");

  assert.strictEqual(estimateTokens(""), 1);
  assert.ok(estimateTokens("abcd") >= 1);
  ok("estimateTokens floors at 1");
}

// --- Unit: tool-call parsing ---
function testToolCallParsing() {
  console.log("parseToolCalls:");

  assert.deepStrictEqual(parseToolCalls("just a normal text answer"), []);
  ok("plain text → no tool calls");

  const clean = parseToolCalls('{"tool_calls":[{"name":"add","arguments":{"a":2,"b":3}}]}');
  assert.strictEqual(clean.length, 1);
  assert.strictEqual(clean[0].function.name, "add");
  assert.deepStrictEqual(clean[0].function.arguments, { a: 2, b: 3 });
  ok("clean tool_calls JSON → normalized {function:{name,arguments}}");

  const fenced = parseToolCalls(
    'Sure, let me do that.\n```json\n{"tool_calls":[{"name":"get_weather","arguments":{"city":"Berlin"}}]}\n```'
  );
  assert.strictEqual(fenced.length, 1);
  assert.strictEqual(fenced[0].function.name, "get_weather");
  assert.deepStrictEqual(fenced[0].function.arguments, { city: "Berlin" });
  ok("tool call inside prose + code fence is extracted");

  const bare = parseToolCalls('{"name":"now","arguments":{}}');
  assert.strictEqual(bare.length, 1);
  assert.strictEqual(bare[0].function.name, "now");
  ok("bare {name,arguments} object accepted");

  const strArgs = parseToolCalls('{"tool_calls":[{"name":"f","arguments":"{\\"x\\":1}"}]}');
  assert.deepStrictEqual(strArgs[0].function.arguments, { x: 1 });
  ok("stringified arguments are parsed into an object");

  const instr = withToolInstruction("What's the weather?", [
    { type: "function", function: { name: "get_weather", description: "d", parameters: {} } },
  ]);
  assert.ok(instr.includes("get_weather"));
  assert.ok(/tool_calls/.test(instr));
  ok("withToolInstruction embeds the tool catalog + protocol");
}

// --- Integration: HTTP endpoints ---
async function testHttp() {
  console.log("HTTP endpoints:");
  const acp = new FakeAcp("Hello from Kiro today.");
  const server = new OllamaServer(acp, { port: 0, host: "127.0.0.1", modelName: "kiro-acp" });
  server.start();
  await new Promise((r) => setTimeout(r, 100));
  const port = server.server.address().port;

  // /api/tags
  const tags = await request(port, "/api/tags", "GET");
  const tagsBody = JSON.parse(tags.body);
  assert.strictEqual(tags.status, 200);
  assert.ok(Array.isArray(tagsBody.models) && tagsBody.models.length === 1);
  assert.strictEqual(tagsBody.models[0].name, "kiro-acp:latest");
  ok("/api/tags lists the bridged model");

  // /api/chat streaming
  const stream = await request(port, "/api/chat", "POST", {
    model: "kiro-acp",
    messages: [{ role: "user", content: "Hi" }],
    stream: true,
  });
  assert.strictEqual(stream.status, 200);
  const lines = stream.body.trim().split("\n").map((l) => JSON.parse(l));
  // Every chunk must carry a message object (ollama ChatResponse requires it).
  for (const l of lines) {
    assert.ok(l.message && typeof l.message.content === "string", "chunk has message.content");
  }
  const last = lines[lines.length - 1];
  assert.strictEqual(last.done, true);
  assert.strictEqual(last.done_reason, "stop");
  assert.strictEqual(typeof last.total_duration, "number");
  assert.strictEqual(typeof last.prompt_eval_count, "number");
  assert.strictEqual(typeof last.eval_count, "number");
  const assembled = lines.map((l) => l.message.content).join("");
  assert.strictEqual(assembled, "Hello from Kiro today.");
  ok("/api/chat streams NDJSON chunks + final metadata; content reassembles");

  // /api/chat non-streaming
  const single = await request(port, "/api/chat", "POST", {
    model: "kiro-acp",
    messages: [{ role: "user", content: "Hi" }],
    stream: false,
  });
  const sBody = JSON.parse(single.body);
  assert.strictEqual(single.status, 200);
  assert.strictEqual(sBody.done, true);
  assert.strictEqual(sBody.message.content, "Hello from Kiro today.");
  assert.strictEqual(typeof sBody.eval_count, "number");
  ok("/api/chat non-streaming returns single JSON with full content");

  // health
  const health = await request(port, "/", "GET");
  assert.strictEqual(health.status, 200);
  assert.ok(/Ollama is running/.test(health.body));
  ok("GET / health check");

  server.stop();
}

// --- Integration: tool-calling mode over HTTP ---
async function testToolHttp() {
  console.log("Tool-calling HTTP flow:");
  // Kiro "decides" to call a function by emitting the directive JSON.
  const acp = new FakeAcp('{"tool_calls":[{"name":"add","arguments":{"a":2,"b":3}}]}');
  const server = new OllamaServer(acp, { port: 0, host: "127.0.0.1", modelName: "kiro-acp" });
  server.start();
  await new Promise((r) => setTimeout(r, 100));
  const port = server.server.address().port;

  const resp = await request(port, "/api/chat", "POST", {
    model: "kiro-acp",
    stream: true,
    messages: [{ role: "user", content: "add 2 and 3" }],
    tools: [
      {
        type: "function",
        function: {
          name: "add",
          description: "Add two numbers",
          parameters: {
            type: "object",
            properties: { a: { type: "number" }, b: { type: "number" } },
          },
        },
      },
    ],
  });
  assert.strictEqual(resp.status, 200);
  const lines = resp.body.trim().split("\n").map((l) => JSON.parse(l));

  // The tool catalog must have reached Kiro (via the prompt).
  assert.ok(/"name": "add"/.test(acp.lastPrompt) || /add/.test(acp.lastPrompt));

  // Some streamed event must carry the tool call in Ollama shape.
  const withCalls = lines.find((l) => l.message && Array.isArray(l.message.tool_calls));
  assert.ok(withCalls, "a chunk carries message.tool_calls");
  assert.strictEqual(withCalls.message.tool_calls[0].function.name, "add");
  assert.deepStrictEqual(withCalls.message.tool_calls[0].function.arguments, { a: 2, b: 3 });

  const last = lines[lines.length - 1];
  assert.strictEqual(last.done, true);
  assert.strictEqual(last.done_reason, "tool_calls");
  ok("/api/chat with tools → streamed tool_calls + done_reason tool_calls");

  server.stop();
}

(async () => {
  try {
    testPromptBuilding();
    testToolCallParsing();
    await testHttp();
    await testToolHttp();
    console.log(`\nAll ${passed} checks passed.`);
    process.exit(0);
  } catch (err) {
    console.error("\nTest failed:", err.message);
    console.error(err.stack);
    process.exit(1);
  }
})();
