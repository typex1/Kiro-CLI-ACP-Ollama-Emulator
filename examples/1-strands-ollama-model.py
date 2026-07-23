from strands import Agent
from strands.models.ollama import OllamaModel

#ollama_model = OllamaModel(
#    host="http://localhost:11434",
#    model_id="gemma4:latest",
#)

ollama_model = OllamaModel(
    host="http://localhost:11435",
    model_id="Kiro",
)

agent = Agent(model=ollama_model)
#agent("Explain the agent loop in one paragraph.")
agent("Which is the latest book written by Stephen King?")