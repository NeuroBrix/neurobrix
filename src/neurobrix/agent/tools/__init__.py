"""Tool registry: a tool = name + JSON schema + handler.

Registry-extensible: any caller may register additional ToolSpecs; v1
ships exactly the six scoped tools (read_file, write_file, edit_file,
list_dir, grep, bash). `schemas_payload()` renders the `tools=` value
the chat templates consume (the standard function-tool shape).
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from neurobrix.agent.sandbox import Sandbox


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema for the arguments object
    handler: Callable[..., str]  # kwargs per schema → text shown to the model

    def schema_payload(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(
                f"unknown tool: {name!r} (available: {sorted(self._tools)})"
            )
        return self._tools[name]

    def names(self) -> List[str]:
        return sorted(self._tools)

    def schemas_payload(self) -> List[Dict[str, Any]]:
        return [self._tools[n].schema_payload() for n in sorted(self._tools)]


def default_toolset(sandbox: Sandbox) -> ToolRegistry:
    """The six scoped v1 tools, bound to one sandbox."""
    from neurobrix.agent.tools.fs import register_fs_tools
    from neurobrix.agent.tools.shell import register_shell_tools

    registry = ToolRegistry()
    register_fs_tools(registry, sandbox)
    register_shell_tools(registry, sandbox)
    return registry
