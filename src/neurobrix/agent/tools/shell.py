"""The bash tool — command execution through the sandbox."""

from neurobrix.agent.sandbox import Sandbox
from neurobrix.agent.tools import ToolRegistry, ToolSpec


def register_shell_tools(registry: ToolRegistry, sandbox: Sandbox) -> None:
    def bash(command: str, timeout: int = 0) -> str:
        rc, output = sandbox.run_command(
            command, timeout=int(timeout) or None
        )
        body = output if output.strip() else "[no output]"
        return f"exit code: {rc}\n{body}"

    registry.register(ToolSpec(
        "bash",
        "Run a shell command in the workdir (captured stdout+stderr; "
        "no network unless the session allows it).",
        {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {
                    "type": "integer",
                    "description": "Seconds; 0 = session default",
                },
            },
            "required": ["command"],
        },
        bash,
    ))
