"""Agent loop for tool-use task execution."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from .questions import QuestionRegistry
from .tools import ToolExecutor


async def run_agent_loop(
    provider: Any,
    model_id: str,
    messages: list[dict],
    prompt: str,
    executor: ToolExecutor,
    max_turns: int = 20,
    max_tokens: int = 4096,
    on_event: Callable[[str, str], None] | None = None,
    question_registry: QuestionRegistry | None = None,
    model_name: str | None = None,
) -> str:
    """Async tool-use agent loop, mutating *messages* in place.

    Appends the user message, all assistant/tool turns, and the final
    assistant response to *messages*.  Returns the final text response.

    When max_turns is exhausted and a question_registry is provided, the
    loop pauses and asks the user for a continuation prompt rather than
    giving up.

    on_event(type, content) is called for observability:
        "chunk"       → streaming text chunk (fires many times per turn)
        "tool_call"   → "{tool_name}: {args_json}"
        "tool_result" → tool output string
        "text"        → final assistant text (full accumulated response)
    """
    messages.append({"role": "user", "content": prompt})

    def _on_chunk(chunk: str) -> None:
        if on_event:
            on_event("chunk", chunk)

    turns_used = 0
    while True:
        for _ in range(max_turns - turns_used):
            turns_used += 1
            msg = await provider.async_stream_raw(
                model_id,
                messages,
                max_tokens,
                tools=executor.tool_definitions,
                on_chunk=_on_chunk,
            )
            tool_calls = msg.get("tool_calls")

            if not tool_calls:
                text = msg.get("content") or ""
                messages.append({"role": "assistant", "content": text})
                if on_event:
                    on_event("text", text)
                return text

            assistant_msg: dict = {"role": "assistant", "tool_calls": tool_calls}
            if msg.get("content"):
                assistant_msg["content"] = msg["content"]
            messages.append(assistant_msg)

            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                tool_id = tc.get("id", "")
                try:
                    arguments = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                if on_event:
                    on_event(
                        "tool_call", f"{tool_name}: {fn.get('arguments', '')}"
                    )

                try:
                    result = await asyncio.to_thread(
                        executor.execute, tool_name, arguments
                    )
                except Exception as e:
                    result = f"Error executing {tool_name}: {e}"

                if on_event:
                    on_event("tool_result", result)

                if not tool_id:
                    tool_id = f"call_{tool_name}_{id(tc)}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result or "(no output)",
                    }
                )

        # Max turns exhausted — ask user whether to continue
        if question_registry and model_name:
            if on_event:
                on_event(
                    "status",
                    f"reached {turns_used} turns — waiting for instructions",
                )
            continuation = await asyncio.to_thread(
                question_registry.ask,
                model_name,
                f"Reached {turns_used} tool turns without a final response. "
                "Send a message to continue, or reply 'stop' to end.",
            )
            if continuation.lower().strip() in ("stop", "quit", "exit"):
                text = f"[stopped by user after {turns_used} turns]"
                messages.append({"role": "assistant", "content": text})
                return text
            messages.append({"role": "user", "content": continuation})
            turns_used = 0
        else:
            text = "[max turns reached without final response]"
            messages.append({"role": "assistant", "content": text})
            return text
