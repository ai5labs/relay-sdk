# Relay AI SDK

Official Python SDK for the [Relay AI Gateway](https://relay.ai5labs.com). One key, every model.

```bash
pip install relay-ai
```

## Quick start

```python
from relay_ai import Relay

client = Relay(api_key="sk-relay-...")

# Chat
response = client.chat("claude-sonnet-4.6", messages=[
    {"role": "user", "content": "Explain quantum computing in one sentence."}
])
print(response.text)

# Streaming
for chunk in client.chat("gemini-3.5-flash", messages=[
    {"role": "user", "content": "Write a haiku about code."}
], stream=True):
    print(chunk.text, end="", flush=True)

# Image generation
result = client.images("flux-schnell", prompt="A cat astronaut on Mars")
print(result.images[0])  # URL
```

## Async

```python
from relay_ai import AsyncRelay

async with AsyncRelay(api_key="sk-relay-...") as client:
    resp = await client.chat("claude-opus-4.8", messages=[
        {"role": "user", "content": "Hello!"}
    ])
    print(resp.text)
```

## Environment variable

Set `RELAY_API_KEY` to skip passing it explicitly:

```bash
export RELAY_API_KEY=sk-relay-...
```

```python
client = Relay()  # picks up from env
```

## Available models

```python
client = Relay()
print(client.models())
```

Chat, image, voice, and video models — all through one API key. See the full list at [relay.ai5labs.com/models](https://relay.ai5labs.com/models).

## Tool calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

resp = client.chat("claude-sonnet-4.6", messages=[
    {"role": "user", "content": "What's the weather in Tokyo?"}
], tools=tools)

print(resp.raw["choices"][0]["message"]["tool_calls"])
```

## License

MIT
