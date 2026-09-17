---
name: google_adk_integration
description: Directives, code patterns, and documentation for working with the Google Agent Development Kit (ADK), Google GenAI SDK, and Google AI Studio.
---

# Google AI Agent Development Guide

This guide provides instructions and reference code for building agents and workflows using Google's official agentic libraries: the Google Agent Development Kit (ADK), Google GenAI SDK, and Google Antigravity SDK.

## 1. Google GenAI SDK

The official SDK for interacting with Gemini models is the `google-genai` package.

### Basic Initialization and Call
```python
from google import genai
from google.genai import types

# Client automatically uses the GEMINI_API_KEY environment variable
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain RAG in one sentence."
)
print(response.text)
```

### Streaming Responses
```python
response = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents="Write a short poem about coding."
)
for chunk in response:
    print(chunk.text, end="", flush=True)
```

---

## 2. Google Agent Development Kit (ADK)

Google ADK is a code-first, framework-agnostic toolkit for building complex multi-agent architectures.

### Creating an Agent
```python
from google.adk import Agent

agent = Agent(
    name="MedicalAssistant",
    model="gemini-2.5-flash",
    instruction="You are a medical assistant. Always tell the user to consult a professional."
)

response = agent.run("I have a mild headache.")
print(response.text)
```

### Integrating Model Context Protocol (MCP) Tools
ADK agents can consume external tools exposed by MCP servers using the `McpToolset`.
```python
from google.adk.tools.mcp_tool import MCPToolset, StdioConnectionParams, StdioServerParameters

mcp_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uv",
            args=["run", "mcp_server.py"]
        )
    )
)

agent = Agent(
    name="ToolAssistant",
    model="gemini-2.5-flash",
    tools=[mcp_toolset],
    instruction="Use the MCP tools to fetch the latest medical records."
)
```

---

## 3. Google Antigravity SDK

The `google-antigravity` Python SDK is used to build autonomous agents that can execute local workspace tools and interact with files or systems safely.

```python
from google.antigravity import AntigravityAgent

agent = AntigravityAgent(
    model="gemini-2.5-flash",
    instructions="Review the workspace files and find any unused dependencies."
)

# Run the agent over a task
result = agent.execute_task("Find unused dependencies in requirements.txt")
print(result)
```

---

## 4. Google AI Studio Credentials

Ensure that you have set the correct environment variables:
- `GEMINI_API_KEY`: Required by the `google-genai` SDK and Google AI Studio.
- `GOOGLE_API_KEY`: Often used by third-party integrations (e.g., LangChain) as a fallback.
