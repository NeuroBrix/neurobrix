"""Agent conversation state and stop conditions."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_SYSTEM_PREAMBLE = """You are a coding agent operating inside a terminal workdir.

Work method:
- Use the available tools to inspect, create, edit and run code until the
  task is complete. Prefer small verifiable steps: read before you edit,
  run tests after you change behavior.
- Every file path is relative to the workdir; you cannot leave it.
- When (and only when) the task is done and verified, reply WITHOUT any
  tool call: that final message is your report to the user."""


@dataclass
class AgentConfig:
    max_turns: int = 25
    malformed_retries: int = 1
    gen_kwargs: Dict[str, Any] = field(default_factory=dict)


class AgentSession:
    """Message list + turn/stop accounting for one agent task."""

    def __init__(self, task: str, workdir: str, system_extra: Optional[str] = None):
        system = _SYSTEM_PREAMBLE + f"\n\nWorkdir: {workdir}"
        if system_extra:
            system += "\n\n" + system_extra
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        self.turns = 0
        self.stop_reason: Optional[str] = None  # final_answer | max_turns | error
        self.final_answer: str = ""

    def add_assistant_turn(self, prose: str, tool_calls: List[Dict[str, Any]]) -> None:
        message: Dict[str, Any] = {"role": "assistant", "content": prose}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.messages.append(message)

    def add_tool_result(self, name: str, content: str) -> None:
        self.messages.append({"role": "tool", "name": name, "content": content})

    def add_user_note(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def stop(self, reason: str, final_answer: str = "") -> None:
        self.stop_reason = reason
        self.final_answer = final_answer
