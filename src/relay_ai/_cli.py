"""CLI for the Relay AI SDK.

Usage::

    relay models                                   # list models
    relay chat claude-sonnet-4.6 "Hello!"          # one-shot chat
    relay chat gemini-3.5-flash "Hi" --stream      # stream tokens
    relay credits                                   # check balance
    relay version                                   # print version
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="relay",
        description="Relay AI — one key, every model",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="Print SDK version")

    p_models = sub.add_parser("models", help="List available models")
    p_models.add_argument("--json", action="store_true", help="JSON output")

    p_chat = sub.add_parser("chat", help="Chat with a model")
    p_chat.add_argument("model", help="Model alias (e.g. claude-sonnet-4.6)")
    p_chat.add_argument("prompt", help="User message")
    p_chat.add_argument("-s", "--stream", action="store_true", help="Stream tokens")
    p_chat.add_argument("--json", action="store_true", help="Print raw JSON response")
    p_chat.add_argument("--max-tokens", type=int, default=None)
    p_chat.add_argument("--temperature", type=float, default=None)

    sub.add_parser("credits", help="Check credit balance")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "version":
        from relay_ai._version import __version__

        print(f"relay-ai {__version__}")
        return 0

    import os

    api_key = os.environ.get("RELAY_API_KEY", "")
    if not api_key:
        print(
            "Error: RELAY_API_KEY is required.\n  export RELAY_API_KEY=sk-relay-...",
            file=sys.stderr,
        )
        return 1

    from relay_ai._client import Relay
    from relay_ai._errors import RelayError

    try:
        with Relay(api_key=api_key, observability=False) as client:
            return _dispatch(args, client)
    except RelayError as exc:
        print(f"Error: {exc.message}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def _dispatch(args: argparse.Namespace, client: object) -> int:
    from relay_ai._client import Relay

    assert isinstance(client, Relay)

    if args.command == "models":
        models = client.models()
        if getattr(args, "json", False):
            print(json.dumps(models, indent=2))
        else:
            for m in models:
                print(m)

    elif args.command == "chat":
        kwargs: dict[str, Any] = {}
        if args.max_tokens is not None:
            kwargs["max_tokens"] = args.max_tokens
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature

        messages = [{"role": "user", "content": args.prompt}]

        if args.stream:
            with client.chat(args.model, messages, stream=True, **kwargs) as stream:  # type: ignore[union-attr]
                for chunk in stream:
                    print(chunk.text, end="", flush=True)
                print()
                if getattr(args, "json", False):
                    final = stream.get_final_response()
                    print(json.dumps(final.model_dump(), indent=2, default=str))
        else:
            resp = client.chat(args.model, messages, **kwargs)
            if getattr(args, "json", False):
                print(json.dumps(resp.raw, indent=2))  # type: ignore[union-attr]
            else:
                print(resp.text)  # type: ignore[union-attr]

    elif args.command == "credits":
        state = client.credits()
        print(f"Balance: ${state.balance_cents / 100:.2f}")
        if state.min_topup_cents or state.max_topup_cents:
            print(
                f"Top-up:  ${state.min_topup_cents / 100:.2f}"
                f" – ${state.max_topup_cents / 100:.2f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
