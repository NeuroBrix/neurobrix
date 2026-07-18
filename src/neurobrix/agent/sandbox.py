"""Path jail and execution policy for agent tool execution.

Every file path a tool touches resolves canonically INSIDE the workdir
(symlinks resolved BEFORE the containment check — escapes are rejected
by construction, not by pattern). `bash` runs with cwd=workdir, a
per-command timeout, and no network by default.

Network cut mechanism, probed once at init:
  1. `unshare -rn` (unprivileged user+net namespaces) when the kernel
     grants it — the command truly has no network.
  2. Degraded fallback: proxy environment poisoned toward an unroutable
     address (blocks well-behaved HTTP(S) tooling). The degraded mode is
     recorded so the transcript carries the truth.
`--allow-network` skips both.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

DEFAULT_BASH_TIMEOUT_S = 120
_OUTPUT_BOUND = 30_000  # chars of combined output surfaced to the model


class SandboxViolation(Exception):
    """A tool attempted something the policy rejects by construction."""


@dataclass(frozen=True)
class SandboxPolicy:
    """Execution policy. Modes: approve_all | default | yolo."""

    mode: str = "default"
    allow_network: bool = False
    bash_timeout_s: int = DEFAULT_BASH_TIMEOUT_S

    def __post_init__(self):
        if self.mode not in ("approve_all", "default", "yolo"):
            raise ValueError(f"Unknown sandbox mode: {self.mode!r}")


class Sandbox:
    def __init__(
        self,
        workdir: str,
        policy: Optional[SandboxPolicy] = None,
        confirm_fn: Optional[Callable[[str], bool]] = None,
    ):
        self.workdir = Path(workdir).resolve()
        if not self.workdir.is_dir():
            raise SandboxViolation(f"workdir does not exist: {self.workdir}")
        self.policy = policy or SandboxPolicy()
        self._confirm_fn = confirm_fn
        if self.policy.mode == "approve_all" and confirm_fn is None:
            raise SandboxViolation(
                "approve_all mode requires an interactive confirmer"
            )
        self.network_isolation = self._probe_network_isolation()

    # ── path jail ────────────────────────────────────────────────────

    def resolve(self, path_str: str) -> Path:
        """Resolve a tool-supplied path inside the jail or raise."""
        candidate = Path(path_str)
        if not candidate.is_absolute():
            candidate = self.workdir / candidate
        resolved = candidate.resolve()
        if resolved != self.workdir and self.workdir not in resolved.parents:
            raise SandboxViolation(
                f"path escapes the workdir jail: {path_str!r} → {resolved}"
            )
        return resolved

    # ── approval policy ──────────────────────────────────────────────

    def approve(self, action: str, detail: str) -> None:
        """Gate one tool execution. Raises SandboxViolation on refusal."""
        if self.policy.mode == "approve_all":
            confirm = self._confirm_fn
            if confirm is None or not confirm(f"{action}: {detail}"):
                raise SandboxViolation(f"user declined: {action}")
        # default / yolo: free inside the jail; refusals are structural
        # (the jail itself, sudo, network) — nothing to confirm.

    # ── command execution ────────────────────────────────────────────

    def run_command(self, command: str, timeout: Optional[int] = None) -> Tuple[int, str]:
        """Run one shell command inside the jail. Returns (rc, output)."""
        if _mentions_sudo(command):
            raise SandboxViolation("sudo is refused inside the agent sandbox")
        self.approve("bash", command)

        argv = ["bash", "-c", command]
        env = dict(os.environ)
        if not self.policy.allow_network:
            if self.network_isolation == "namespace":
                argv = ["unshare", "-rn", "--"] + argv
            else:  # degraded: poison proxies toward an unroutable address
                env.update(
                    http_proxy="http://127.0.0.1:9",
                    https_proxy="http://127.0.0.1:9",
                    HTTP_PROXY="http://127.0.0.1:9",
                    HTTPS_PROXY="http://127.0.0.1:9",
                    no_proxy="",
                    NO_PROXY="",
                )
        try:
            proc = subprocess.run(
                argv,
                cwd=str(self.workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout or self.policy.bash_timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            partial = _as_text(exc.stdout) + _as_text(exc.stderr)
            return 124, _bound(
                f"[timeout after {timeout or self.policy.bash_timeout_s}s]\n{partial}"
            )
        output = proc.stdout + (("\n" + proc.stderr) if proc.stderr else "")
        return proc.returncode, _bound(output)

    # ── internals ────────────────────────────────────────────────────

    def _probe_network_isolation(self) -> str:
        """'namespace' | 'proxy-poison' | 'off' — probed once."""
        if self.policy.allow_network:
            return "off"
        try:
            probe = subprocess.run(
                ["unshare", "-rn", "--", "true"],
                capture_output=True,
                timeout=5,
            )
            if probe.returncode == 0:
                return "namespace"
        except (OSError, subprocess.TimeoutExpired):
            pass
        return "proxy-poison"


def _mentions_sudo(command: str) -> bool:
    import shlex

    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    return "sudo" in tokens


def _as_text(chunk) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, bytes):
        return chunk.decode(errors="replace")
    return chunk


def _bound(text: str, limit: int = _OUTPUT_BOUND) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... output truncated at {limit} chars]"
