"""NeuroBrix agent layer — terminal coding agent above inference.

PURE STDLIB by maintainer decision: this package imports nothing outside
the standard library (mechanically enforced by the purity gate test).
Engine access is dependency-inverted through `complete_fn`; the adapters
live in the CLI layer.
"""

from typing import Any, Callable, Dict, Optional

from neurobrix.agent.loop import AgentLoop, AgentResult
from neurobrix.agent.parsers import select_parser
from neurobrix.agent.sandbox import Sandbox, SandboxPolicy, SandboxViolation
from neurobrix.agent.session import AgentConfig, AgentSession
from neurobrix.agent.tools import ToolRegistry, ToolSpec, default_toolset
from neurobrix.agent.transcript import Transcript

__all__ = [
    "AgentConfig",
    "AgentLoop",
    "AgentResult",
    "AgentSession",
    "Sandbox",
    "SandboxPolicy",
    "SandboxViolation",
    "ToolRegistry",
    "ToolSpec",
    "Transcript",
    "default_toolset",
    "run_agent_task",
    "select_parser",
]


def run_agent_task(
    complete_fn: Callable[..., str],
    task: str,
    workdir: str,
    template_text: str,
    transcript_dir: str,
    policy: Optional[SandboxPolicy] = None,
    config: Optional[AgentConfig] = None,
    confirm_fn: Optional[Callable[[str], bool]] = None,
    meta: Optional[Dict[str, Any]] = None,
    echo: bool = False,
) -> AgentResult:
    """Assemble and run one agent task. The single high-level entry point."""
    parser = select_parser(template_text)
    if parser is None:
        raise ValueError(
            "this model's chat template declares no tool contract — "
            "agent mode is unavailable for it"
        )
    sandbox = Sandbox(workdir, policy=policy, confirm_fn=confirm_fn)
    tools = default_toolset(sandbox)
    session = AgentSession(task, workdir=str(sandbox.workdir))
    transcript = Transcript(
        transcript_dir,
        echo=echo,
        meta={
            "task": task,
            "workdir": str(sandbox.workdir),
            "parser": parser.format_name,
            "network_isolation": sandbox.network_isolation,
            **(meta or {}),
        },
    )
    loop = AgentLoop(
        complete_fn=complete_fn,
        parser=parser,
        tools=tools,
        session=session,
        transcript=transcript,
        config=config or AgentConfig(),
    )
    return loop.run()
