# Relay AI SDK

Official Python SDK for the [Relay AI Gateway](https://relay.ai5labs.com). One key, every model.

```bash
pip install ai5labs-relay
```

With OpenTelemetry:

```bash
pip install ai5labs-relay[otel]
```

## Quick start

```python
from relay_ai import Relay

client = Relay(api_key="sk-relay-...")

response = client.chat("claude-sonnet-4.6", messages=[
    {"role": "user", "content": "Explain quantum computing in one sentence."}
])
print(response.text)
print(f"Tokens: {response.usage.total_tokens}")
```

## Streaming

```python
with client.chat("gemini-3.5-flash", messages=[
    {"role": "user", "content": "Write a haiku about code."}
], stream=True) as stream:
    for chunk in stream:
        print(chunk.text, end="", flush=True)

    final = stream.get_final_response()
    print(f"\nTokens: {final.usage.total_tokens}")
```

## Async

```python
from relay_ai import AsyncRelay

async with AsyncRelay() as client:
    response = await client.chat("claude-opus-4.8", messages=[
        {"role": "user", "content": "Hello!"}
    ])
    print(response.text)
```

## Tool calling

```python
tools = [{
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
}]

response = client.chat("claude-sonnet-4.6", messages=[
    {"role": "user", "content": "What's the weather in Tokyo?"}
], tools=tools)

for tc in response.tool_calls:
    print(f"{tc.function_name}({tc.function_arguments})")
```

## Image generation

```python
result = client.images("flux-schnell", prompt="A cat astronaut on Mars")
print(result.images[0])
```

## Audio

```python
# Transcription
transcript = client.transcribe("whisper-1", "meeting.mp3")
print(transcript.text)

# Text-to-speech
audio = client.speech("tts-1", "Hello from Relay!")
with open("output.mp3", "wb") as f:
    f.write(audio.audio)
```

## Semantic routing

```python
decision = client.route(
    messages=[{"role": "user", "content": "Prove the Riemann hypothesis"}],
    candidates=["claude-opus-4.8", "claude-sonnet-4.6", "gemini-3.5-flash"],
)
print(f"Best model: {decision.alias} ({decision.confidence:.0%})")
print(f"Reasoning: {decision.reasoning}")
```

## Batch processing

```python
results = client.batch("claude-sonnet-4.6", [
    {"messages": [{"role": "user", "content": "What is 2+2?"}]},
    {"messages": [{"role": "user", "content": "What is 3+3?"}]},
    {"messages": [{"role": "user", "content": "What is 4+4?"}]},
], max_concurrent=5)

for r in results:
    if r.response:
        print(f"[{r.index}] {r.response.text}")
    else:
        print(f"[{r.index}] Error: {r.error}")
```

## Credits

```python
state = client.credits()
print(f"Balance: ${state.balance_cents / 100:.2f}")
```

## Error handling

```python
from relay_ai import (
    RelayError,
    AuthenticationError,
    RateLimitError,
    InsufficientCreditsError,
    ModelNotFoundError,
)

try:
    response = client.chat("gpt-5", messages=[...])
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except InsufficientCreditsError:
    print("Top up your credits at relay.ai5labs.com")
except ModelNotFoundError:
    print("Model not found")
except RelayError as e:
    print(f"Error: {e.message}")
```

## CLI

```bash
export RELAY_API_KEY=sk-relay-...

relay models                                # List models
relay chat claude-sonnet-4.6 "Hello!"       # Quick chat
relay chat gemini-3.5-flash "Hi" --stream   # Stream tokens
relay credits                               # Check balance
relay version                               # SDK version
```

## Configuration

```python
client = Relay(
    api_key="sk-relay-...",       # or set RELAY_API_KEY env var
    base_url="https://...",       # custom gateway URL
    timeout=120.0,                # request timeout (seconds)
    max_retries=2,                # automatic retries on 429/5xx
    send_telemetry=True,          # usage analytics (metadata only)
    http_client=httpx.Client(),   # custom httpx client
)
```

## Telemetry

The SDK sends anonymous usage metadata (model, token counts, latency) to improve the service. **No message content, prompts, responses, or tool arguments are ever transmitted.** This is enforced by a client-side allowlist and verified by server-side stripping.

Disable with:

```python
client = Relay(send_telemetry=False)
```

## OpenTelemetry

```python
from relay_ai import Relay
from relay_ai._otel import instrument, RelaySpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(
        RelaySpanExporter(api_key="sk-relay-...", base_url="https://api.relay.ai5labs.com/v1")
    )
)

client = instrument(Relay())
response = client.chat(...)  # Automatically creates OTel spans
```

## License

MIT
