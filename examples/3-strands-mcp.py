import os
from mcp import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.tools.mcp import MCPClient
from strands.models.ollama import OllamaModel

SYSTEM_PROMPT = """You are an AWS Solutions Architect. You help users design and build 
well-architected solutions on AWS. Use the available tools to provide accurate, up-to-date 
guidance on AWS services, architecture patterns, and pricing. When users ask about costs, 
use the pricing tools to give real numbers. Always cite relevant documentation when possible."""

# Remote MCP server (Streamable HTTP) — AWS docs, best practices, architecture guidance
aws_knowledge = MCPClient(
    lambda: streamablehttp_client("https://knowledge-mcp.global.api.aws"),
    prefix="knowledge"
)

ollama_model = OllamaModel(
    host="http://localhost:11435",
    model_id="Claude Opus 4.8",
)

agent = Agent(model=ollama_model, tools=[aws_knowledge], system_prompt=SYSTEM_PROMPT)

print("AWS Architect Agent (type 'quit' to exit)")
print("-" * 45)

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break
    if not user_input:
        continue
    print()
    agent(user_input)