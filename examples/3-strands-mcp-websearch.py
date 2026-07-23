import os
from mcp import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent, tool
from strands.tools.mcp import MCPClient
from strands.models.ollama import OllamaModel
from ddgs import DDGS

SYSTEM_PROMPT = """You are an AWS Solutions Architect. You help users design and build 
well-architected solutions on AWS. Use the available tools to provide accurate, up-to-date 
guidance on AWS services, architecture patterns, and pricing. Always cite relevant 
documentation when possible."""

# Remote MCP server (Streamable HTTP) — AWS docs, best practices, architecture guidance
aws_knowledge = MCPClient(
    lambda: streamablehttp_client("https://knowledge-mcp.global.api.aws"),
    prefix="knowledge"
)

# --- Web search tool using DuckDuckGo (ddgs) ---
@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo and return results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default: 3).

    Returns:
        Search results with titles, URLs, and snippets.
    """
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No results found."
        output = []
        for r in results:
            output.append(f"• {r['title']}\n  {r['href']}\n  {r['body']}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Search error: {e}"


ollama_model = OllamaModel(
    host="http://localhost:11435",
    model_id="Kiro",
)

agent = Agent(model=ollama_model, tools=[aws_knowledge, web_search], system_prompt=SYSTEM_PROMPT)

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