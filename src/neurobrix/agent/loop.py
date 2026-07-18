"""The agent turn loop.

Per turn: complete → parse → execute tools → reinject results → repeat,
until the model answers without a tool call (final answer) or a bound
fires. The loop is ENGINE-BLIND: `complete_fn` is an injected callable
`(messages, tools_payload, **gen_kwargs) → raw assistant text`; the two
adapters (serving daemon / in-process engine) live in the CLI layer.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict

from neurobrix.agent.parsers.base import ToolCallParser
from neurobrix.agent.sandbox import SandboxViolation
from neurobrix.agent.session import AgentConfig, AgentSession
from neurobrix.agent.tools import ToolRegistry
from neurobrix.agent.transcript import Transcript

CompleteFn = Callable[..., str]

_FORMAT_REMINDER = (
    "Your last tool call was malformed and was NOT executed. Re-emit it "
    "exactly in the tool-call format your instructions define, or answer "
    "without a tool call if you are done."
)


@dataclass(frozen=True)
class AgentResult:
    final_answer: str
    stop_reason: str
    turns: int


class AgentLoop:
    def __init__(
        self,
        complete_fn: CompleteFn,
        parser: ToolCallParser,
        tools: ToolRegistry,
        session: AgentSession,
        transcript: Transcript,
        config: AgentConfig,
    ):
        self._complete = complete_fn
        self._parser = parser
        self._tools = tools
        self._session = session
        self._transcript = transcript
        self._config = config
        self._tools_payload = tools.schemas_payload()

    def run(self) -> AgentResult:
        session, transcript = self._session, self._transcript
        malformed_left = self._config.malformed_retries

        while session.stop_reason is None:
            if session.turns >= self._config.max_turns:
                session.stop("max_turns", session.final_answer)
                break
            session.turns += 1
            transcript.turn(session.turns)

            text = self._complete(
                session.messages, self._tools_payload, **self._config.gen_kwargs
            )
            transcript.model_text(text)
            parsed = self._parser.parse(text)

            # Per-turn decode outcome in the machine record: this is the
            # stop-token evidence of the R29 transcript — every turn must
            # end as a complete tool-call block, a clean final answer, or
            # be flagged malformed (dangling opener = truncated/run-past).
            transcript.event(
                "turn_outcome",
                index=session.turns,
                outcome=(
                    f"tool_calls:{len(parsed.calls)}" if parsed.calls
                    else "malformed" if parsed.malformed
                    else "final_answer"
                ),
                chars=len(text),
            )

            if parsed.calls:
                session.add_assistant_turn(
                    parsed.prose,
                    [
                        {
                            "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments},
                        }
                        for c in parsed.calls
                    ],
                )
                for call in parsed.calls:
                    transcript.tool_call(call.name, call.arguments)
                    session.add_tool_result(call.name, self._execute(call))
                malformed_left = self._config.malformed_retries
                continue

            if parsed.malformed:
                if malformed_left > 0:
                    malformed_left -= 1
                    transcript.note("malformed tool call — asking for the format once")
                    session.add_assistant_turn(text, [])
                    session.add_user_note(_FORMAT_REMINDER)
                    continue
                session.stop("error", "unrecoverable malformed tool call")
                break

            # No calls, nothing malformed: the model is done.
            session.add_assistant_turn(parsed.prose, [])
            session.stop("final_answer", parsed.prose)

        transcript.finish(
            session.stop_reason or "error", session.final_answer, session.turns
        )
        return AgentResult(session.final_answer, session.stop_reason or "error", session.turns)

    def _execute(self, call) -> str:
        started = time.monotonic()
        try:
            spec = self._tools.get(call.name)
            missing = [
                key
                for key in spec.parameters.get("required", [])
                if key not in call.arguments
            ]
            if missing:
                output = f"ERROR: missing required argument(s): {', '.join(missing)}"
            else:
                known = set(spec.parameters.get("properties", {}))
                kwargs: Dict[str, Any] = {
                    k: v for k, v in call.arguments.items() if k in known
                }
                output = spec.handler(**kwargs)
        except SandboxViolation as exc:
            output = f"REFUSED by sandbox policy: {exc}"
        except KeyError as exc:
            output = f"ERROR: {exc.args[0]}"
        except Exception as exc:  # surfaced to the model, never swallowed
            output = f"ERROR: {type(exc).__name__}: {exc}"
        self._transcript.tool_result(call.name, output, time.monotonic() - started)
        return output
