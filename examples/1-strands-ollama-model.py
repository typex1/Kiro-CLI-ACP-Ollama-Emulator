from strands import Agent
from strands.models.ollama import OllamaModel


# bedrock = BedrockModel(
#     model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
#     temperature=0.3,
#     top_p=0.8,
# )

# anthropic_model = AnthropicModel(
#     client_args={"api_key": os.environ["ANTHROPIC_API_KEY"]},
#     model_id="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     params={"temperature": 0.7},
# )


# openai_model = OpenAIModel(
#     client_args={"api_key": os.environ["OPENAI_API_KEY"]},
#     model_id="gpt-4o",
#     params={"max_tokens": 1000, "temperature": 0.7},
# )

#ollama_model = OllamaModel(
#    host="http://localhost:11434",
#    model_id="gemma4:latest",
#)

ollama_model = OllamaModel(
    host="http://localhost:11435",
    model_id="Claude Opus 4.8",
)

agent = Agent(model=ollama_model)
#agent("Explain the agent loop in one paragraph.")
agent("Which is the latest book written by Stephen King?")