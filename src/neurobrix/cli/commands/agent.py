"""Agent-mode CLI wiring.

The agent layer (neurobrix.agent) is engine-blind and stdlib-pure; this
module is where the two worlds meet: it builds the `complete_fn` adapter
(warm daemon path or cold in-process engine), reads the embedded chat
template for data-driven parser selection, and assembles the sandbox
policy from CLI flags. Consumed by `neurobrix chat --agent` (REPL) and
`neurobrix run --mode agent` (one-shot).
"""

import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple


def _policy_from_args(args):
    from neurobrix.agent import SandboxPolicy

    if getattr(args, "approve_all", False) and getattr(args, "yolo", False):
        print("ERROR: --approve-all and --yolo are mutually exclusive.")
        sys.exit(1)
    mode = (
        "approve_all"
        if getattr(args, "approve_all", False)
        else "yolo" if getattr(args, "yolo", False) else "default"
    )
    kwargs = {"mode": mode, "allow_network": bool(getattr(args, "allow_network", False))}
    if getattr(args, "bash_timeout", None):
        kwargs["bash_timeout_s"] = int(args.bash_timeout)
    return SandboxPolicy(**kwargs)


def _config_from_args(args, agent_defaults: dict):
    from neurobrix.agent import AgentConfig

    max_turns = getattr(args, "max_turns", None) or int(
        agent_defaults.get("max_turns", 25)
    )
    gen_kwargs = {}
    if getattr(args, "temperature", None) is not None:
        gen_kwargs["temperature"] = args.temperature
    if getattr(args, "max_tokens", None) is not None:
        gen_kwargs["max_tokens"] = args.max_tokens
    return AgentConfig(max_turns=max_turns, gen_kwargs=gen_kwargs)


def _supported_modes(family: str) -> list:
    from neurobrix.core.config import get_family_config

    try:
        cfg = get_family_config(family)
    except Exception:
        return []
    return (cfg.get("modes") or {}).get("supported", [])


def build_warm_adapter(expected_model: Optional[str] = None
                       ) -> Tuple[Callable[..., str], str, str]:
    """Adapter over a running daemon. Returns (complete_fn, template, label)."""
    from neurobrix.serving.client import DaemonClient

    client = DaemonClient()
    client.connect()
    status = client.status()
    if expected_model and status.get("model") != expected_model:
        print(f"ERROR: daemon serves '{status.get('model')}' but --model "
              f"'{expected_model}' was requested. Stop the daemon "
              f"(neurobrix stop) or drop --model to use the warm path.")
        sys.exit(1)
    family = status.get("family") or ""
    if "agent" not in _supported_modes(family):
        print(f"ERROR: family '{family}' does not support agent mode "
              f"(modes.supported in its family YAML).")
        sys.exit(1)
    template = client.template().get("text", "")

    def complete_fn(messages, tools_payload, **gen) -> str:
        return client.complete(messages, tools=tools_payload, **gen).get("text", "")

    label = f"daemon:{status.get('model')}:{status.get('mode')}"
    return complete_fn, template, label


def build_cold_adapter(model: str, hardware: Optional[str], engine_mode: str
                       ) -> Tuple[Callable[..., str], str, str]:
    """In-process engine adapter. Returns (complete_fn, template, label)."""
    from neurobrix.serving.engine import InferenceEngine

    # Gate BEFORE the expensive engine load: the family is a cheap manifest
    # read, and the predicate is the same data-driven modes.supported check
    # the warm adapter applies (doctrine-review absorption 2026-07-18).
    from neurobrix.nbx import NBXContainer
    from neurobrix.cli.utils import find_model

    manifest = NBXContainer.load(str(find_model(model))).get_manifest() or {}
    family = manifest.get("family") or ""
    if "agent" not in _supported_modes(family):
        print(f"ERROR: family '{family}' does not support agent mode "
              f"(modes.supported in its family YAML).")
        sys.exit(1)

    if not hardware:
        from neurobrix.core.prism.autodetect import get_or_create_default_profile

        hardware = get_or_create_default_profile()
    engine = InferenceEngine(model, hardware_id=hardware, mode=engine_mode)
    engine.load()
    template = engine.chat_template_text()

    def complete_fn(messages, tools_payload, **gen) -> str:
        return engine.complete(messages, tools=tools_payload, **gen)

    label = f"cold:{model}:{engine_mode}"
    return complete_fn, template, label


def _transcript_dir(args, workdir: Path) -> str:
    explicit = getattr(args, "transcript_dir", None)
    if explicit:
        return explicit
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return str(workdir / ".nbx_agent" / stamp)


def run_agent_once(args, task: str, complete_fn, template: str, label: str,
                   agent_defaults: dict, echo: bool) -> int:
    """One agent task through the pure loop. Returns exit code."""
    from neurobrix.agent import SandboxViolation, run_agent_task

    workdir = Path(getattr(args, "workdir", None) or ".").resolve()
    confirm_fn = None
    if getattr(args, "approve_all", False):
        confirm_fn = lambda prompt: input(f"[approve] {prompt} — y/N: ").strip().lower() == "y"

    try:
        result = run_agent_task(
            complete_fn=complete_fn,
            task=task,
            workdir=str(workdir),
            template_text=template,
            transcript_dir=_transcript_dir(args, workdir),
            policy=_policy_from_args(args),
            config=_config_from_args(args, agent_defaults),
            confirm_fn=confirm_fn,
            meta={"backend": label},
            echo=echo,
        )
    except (ValueError, SandboxViolation) as exc:
        print(f"ERROR: {exc}")
        return 1

    if not echo:
        print(result.final_answer)
    print(f"\n[Agent] {result.stop_reason} after {result.turns} turn(s)")

    output = getattr(args, "output", None)
    if output:
        Path(output).write_text(result.final_answer + "\n")
        print(f"[Agent] Final answer saved to: {output}")
    return 0 if result.stop_reason == "final_answer" else 1


def agent_defaults() -> dict:
    from neurobrix.core.config import get_family_config

    return get_family_config("llm").get("agent", {}) or {}


def _engine_mode_from_args(args) -> str:
    if getattr(args, "sequential", False):
        return "sequential"
    if getattr(args, "triton_sequential", False):
        return "triton_sequential"
    if getattr(args, "triton", False):
        return "triton"
    return "compiled"


def run_agent_mode(args) -> int:
    """One-shot agent task: neurobrix run --mode agent --prompt "<task>"."""
    from neurobrix.serving.client import DaemonClient

    task = getattr(args, "prompt", None)
    if not task:
        print("ERROR: --mode agent requires --prompt \"<task>\".")
        return 1

    if DaemonClient.is_running():
        complete_fn, template, label = build_warm_adapter(
            expected_model=getattr(args, "model", None))
        print(f"[Agent] Warm path — {label}")
    else:
        engine_mode = _engine_mode_from_args(args)
        print(f"[Agent] Cold path — loading {args.model} ({engine_mode})")
        complete_fn, template, label = build_cold_adapter(
            args.model, getattr(args, "hardware", None), engine_mode
        )

    return run_agent_once(args, task, complete_fn, template, label,
                          agent_defaults(), echo=True)


def cmd_agent_repl(args, agent_defaults: dict) -> None:
    """Interactive agent REPL over the daemon (chat --agent)."""
    from neurobrix.serving.client import DaemonClient

    if not DaemonClient.is_running():
        print("[Agent] No daemon running. Start one first: neurobrix serve --model <name>")
        sys.exit(1)
    complete_fn, template, label = build_warm_adapter(
        expected_model=getattr(args, "model", None))

    print(f"\nNeuroBrix Agent — {label}")
    print("Type a task; the agent works inside the workdir jail. /quit to exit.")
    print(f"Workdir: {Path(getattr(args, 'workdir', None) or '.').resolve()}")
    print("─" * 50)
    while True:
        try:
            task = input("\nTask: ").strip()
        except EOFError:
            break
        if not task:
            continue
        if task.lower() in ("/quit", "/exit", "/q"):
            break
        run_agent_once(args, task, complete_fn, template, label,
                       agent_defaults, echo=True)
