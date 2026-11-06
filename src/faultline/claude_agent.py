"""A real coding agent as a baseline: Claude, driving the environment through tool use.

Needs the optional `agents` dependencies and Anthropic credentials.
"""

from __future__ import annotations

from typing import Any, cast

import anthropic

from faultline.env import FaultlineEnv
from faultline.models import ActionKind

DEFAULT_MODEL = "claude-opus-5-5"
_MAX_OUTPUT_TOKENS = 16_000
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_SYSTEM = """You are repairing a production order service. The service is a FastAPI app with a \
transactional outbox and delivery workers, backed by SQLite, that calls an external payment and \
shipping provider.

A hidden verifier will score your final code against scenarios you cannot see. It checks real \
behavior through the provider's own records, including crashes, retries, slow responses and old \
data. The visible tests may already pass while the service is still broken, so reason about \
correctness and do not stop at green tests. Reporting status without doing the work, disabling \
features, or changing tests will fail.

Investigate with the tools, fix the root causes, and call submit when you are confident. You have \
a limited number of tool calls, and every call uses one. Submit before you run out."""


def _tool(name: ActionKind, description: str, argument: str | None = None) -> dict[str, Any]:
    properties = {"argument": {"type": "string", "description": argument}} if argument else {}
    return {
        "name": name.value,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


TOOLS: Any = [
    _tool(
        ActionKind.READ,
        "Read a workspace file with line numbers.",
        "path, path:line or path:first-last",
    ),
    _tool(
        ActionKind.SEARCH,
        "Search all workspace files for exact text. Empty text lists every file.",
        "text to find",
    ),
    _tool(
        ActionKind.RUN,
        "Run the visible tests.",
        "pytest -q, optionally followed by a path under tests/",
    ),
    _tool(
        ActionKind.LOGS,
        "Read recent logs and the provider's records.",
        "all, api, worker or provider",
    ),
    _tool(
        ActionKind.QUERY,
        "Run a read-only SQL query against the live database.",
        "one SELECT statement",
    ),
    _tool(
        ActionKind.PATCH,
        "Replace text that occurs exactly once in a file.",
        'JSON object with string fields "path", "old" and "new"',
    ),
    _tool(
        ActionKind.RESTART,
        "Restart the live service on the current code and resend the incident traffic.",
    ),
    _tool(ActionKind.CHECKPOINT, "Save the current files and data. Returns a checkpoint ID."),
    _tool(
        ActionKind.RESTORE,
        "Return files and data to a checkpoint. The tool budget keeps counting.",
        "checkpoint ID",
    ),
    _tool(
        ActionKind.SUBMIT, "Submit the current code for final verification. This ends the episode."
    ),
]


def _status(observation: dict[str, Any]) -> str:
    return (
        f"[steps left: {int(observation['remaining_steps'])}; "
        f"services: {observation['service_summary']}; "
        f"changes: {observation['diff_summary']}; "
        f"last tests: {observation['visible_test_summary']}]"
    )


def make_claude_policy(
    model: str = DEFAULT_MODEL,
    effort: str = "high",
    client: anthropic.Anthropic | None = None,
    usage: dict[str, int] | None = None,
) -> Any:
    api = client or anthropic.Anthropic()

    def policy(_: FaultlineEnv, observation: dict[str, Any], step: Any) -> None:
        messages: Any = [
            {
                "role": "user",
                "content": f"Issue report:\n{observation['issue']}\n\n{_status(observation)}",
            }
        ]
        done = False
        while not done:
            with api.beta.messages.stream(
                model=model,
                max_tokens=_MAX_OUTPUT_TOKENS,
                system=_SYSTEM,
                tools=TOOLS,
                messages=messages,
                output_config=cast(Any, {"effort": effort}),
                cache_control={"type": "ephemeral"},
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                response = stream.get_final_message()
            if usage is not None:
                usage["input"] = usage.get("input", 0) + response.usage.input_tokens
                usage["output"] = usage.get("output", 0) + response.usage.output_tokens
                usage["cache_read"] = usage.get("cache_read", 0) + (
                    response.usage.cache_read_input_tokens or 0
                )
                usage["cache_write"] = usage.get("cache_write", 0) + (
                    response.usage.cache_creation_input_tokens or 0
                )
            # Replay the assistant turn unchanged: thinking blocks are bound to this history.
            messages.append({"role": "assistant", "content": response.content})
            calls = [block for block in response.content if block.type == "tool_use"]
            if response.stop_reason == "refusal" or (
                not calls and response.stop_reason != "max_tokens"
            ):
                return
            results: list[dict[str, Any]] = []
            for call in calls:
                if done:
                    # The episode ended on an earlier call in this turn.
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": "not run: the episode has ended",
                            "is_error": True,
                        }
                    )
                    continue
                observation, done = step(ActionKind(call.name), call.input.get("argument", ""))
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": f"{observation['last_result']}\n\n{_status(observation)}",
                    }
                )
            if results:
                messages.append({"role": "user", "content": results})
            elif not done:
                messages.append({"role": "user", "content": "Continue."})

    return policy
