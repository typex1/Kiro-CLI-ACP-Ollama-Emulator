/**
 * Kiro ACP → Ollama Emulation Server — Entry Point
 *
 * Exposes Kiro CLI (via ACP) as a local Ollama-compatible HTTP server.
 * Point any Ollama client (e.g. the Strands Agents `OllamaModel` provider) at
 * http://localhost:11435 and it will talk to Kiro instead of a real model.
 *
 * Usage:
 *   1. (Optional) cp .env_example .env and configure
 *   2. Authenticate Kiro once:  kiro-cli chat
 *   3. npm start
 *   4. Point your Strands agent at host="http://localhost:11435"
 */

const AcpClient = require("./acp-client");
const OllamaServer = require("./ollama-server");

// Load .env (simple parser; no external dependency)
try {
  const fs = require("fs");
  const envPath = require("path").join(__dirname, "..", ".env");
  if (fs.existsSync(envPath)) {
    for (const line of fs.readFileSync(envPath, "utf-8").split("\n")) {
      const [key, ...val] = line.split("=");
      if (key && !key.startsWith("#"))
        process.env[key.trim()] = val.join("=").trim();
    }
  }
} catch (_) {}

const PORT = Number(process.env.OLLAMA_PORT || 11435);
const HOST = process.env.OLLAMA_HOST || "127.0.0.1";
const MODEL_NAME = process.env.OLLAMA_MODEL_NAME || "kiro-acp";
// Tool-call translation is on unless explicitly disabled.
const TOOL_CALLS = !/^(off|false|0|no)$/i.test(process.env.OLLAMA_TOOLCALLS || "");

async function main() {
  console.log("🌉 Kiro ACP → Ollama emulation server starting...");
  console.log(`   Workspace: ${process.env.KIRO_WORKSPACE || process.cwd()}`);

  const acp = new AcpClient();
  await acp.start();
  console.log("✅ Kiro ACP connected");

  const server = new OllamaServer(acp, {
    port: PORT,
    host: HOST,
    modelName: MODEL_NAME,
    toolCalls: TOOL_CALLS,
  });
  server.start();
  console.log("");
  console.log(`🚀 Ready. Set your Ollama client host to http://${HOST}:${PORT}`);
  console.log("   The model_id you request is ignored — every call routes to Kiro.\n");

  // If Kiro exits, we can't serve anything — shut down cleanly.
  acp.on("exit", () => {
    console.error("Kiro CLI process ended. Shutting down server.");
    server.stop();
    process.exit(1);
  });

  const shutdown = () => {
    console.log("\nShutting down...");
    server.stop();
    acp.stop();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
