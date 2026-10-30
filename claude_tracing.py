"""Chat with Claude and trace each turn to Elastic's Managed OTLP endpoint."""

import argparse
import os
import sys
from uuid import uuid4

from anthropic import APIStatusError, Anthropic
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


def api_error_message(error: APIStatusError) -> str:
    """Return the provider's own explanation, without the raw response body."""
    body = error.body
    if isinstance(body, dict):
        detail = body.get("error")
        if isinstance(detail, dict):
            message = detail.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
    return error.message


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Optional first message")
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--console", action="store_true", help="Print spans locally instead of sending to Elastic")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        parser.error("set ANTHROPIC_API_KEY before running")
    if not args.console:
        if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            parser.error("set OTEL_EXPORTER_OTLP_ENDPOINT to your Elastic Managed OTLP base URL")
        if not os.getenv("OTEL_EXPORTER_OTLP_HEADERS"):
            parser.error("set OTEL_EXPORTER_OTLP_HEADERS to 'Authorization=ApiKey <encoded-key>'")

    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "claude-tracing-test"}))
    exporter = ConsoleSpanExporter() if args.console else OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    try:
        # Record prompt and response content on the SDK span.
        # Use "content_mode": "metadata_only" to not record prompt and response content.
        client = Anthropic(
            open_telemetry={
                "tracer_provider": provider,
                "traces": {"content_mode": "content"},
            }
        )
        conversation_id = f"chat-{uuid4()}"
        client = client.with_options(
            open_telemetry={**client.open_telemetry, "conversation_id": conversation_id}
        )
        history = []
        next_prompt = args.prompt
        print("Chat with Claude. Type /exit to quit.")
        while True:
            if next_prompt is None:
                try:
                    next_prompt = input("You: ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
            prompt = next_prompt.strip()
            next_prompt = None
            if prompt.lower() in {"/exit", "/quit"}:
                break
            if not prompt:
                continue

            turn = [*history, {"role": "user", "content": prompt}]
            with trace.get_tracer(__name__).start_as_current_span("claude-chat-turn") as span:
                trace_id = f"{span.get_span_context().trace_id:032x}"
                try:
                    message = client.messages.create(
                        model=args.model,
                        max_tokens=1024,
                        messages=turn,
                    )
                except APIStatusError as error:
                    message_text = api_error_message(error)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, message_text))
                    print(f"Claude API error: {message_text}", file=sys.stderr)
                    if error.request_id:
                        print(f"Request ID: {error.request_id}", file=sys.stderr)
                    return 1
            history = [*turn, {"role": "assistant", "content": message.content}]
            answer = "\n".join(block.text for block in message.content if block.type == "text")
            print(f"Claude: {answer}")
            print(f"Trace ID: {trace_id}", file=sys.stderr)
            provider.force_flush(timeout_millis=10000)
        return 0
    finally:
        provider.force_flush(timeout_millis=10000)
        provider.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
