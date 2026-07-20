"""Sandbox tests — jail escapes rejected by construction, policy refusals,
timeout, and output bounding."""

import os

import pytest

from neurobrix.agent.sandbox import Sandbox, SandboxPolicy, SandboxViolation


@pytest.fixture()
def sandbox(tmp_path):
    return Sandbox(str(tmp_path), SandboxPolicy(mode="yolo"))


def test_relative_and_absolute_paths_resolve_inside(sandbox, tmp_path):
    assert sandbox.resolve("a/b.txt") == tmp_path.resolve() / "a" / "b.txt"
    assert sandbox.resolve(str(tmp_path / "c.txt")) == tmp_path.resolve() / "c.txt"


def test_dotdot_escape_rejected(sandbox):
    with pytest.raises(SandboxViolation):
        sandbox.resolve("../outside.txt")
    with pytest.raises(SandboxViolation):
        sandbox.resolve("a/../../outside.txt")


def test_absolute_escape_rejected(sandbox):
    with pytest.raises(SandboxViolation):
        sandbox.resolve("/etc/passwd")


def test_symlink_escape_rejected(sandbox, tmp_path):
    link = tmp_path / "sneaky"
    os.symlink("/etc", link)
    with pytest.raises(SandboxViolation):
        sandbox.resolve("sneaky/passwd")


def test_sudo_refused(sandbox):
    with pytest.raises(SandboxViolation):
        sandbox.run_command("sudo whoami")


def test_command_runs_in_workdir(sandbox, tmp_path):
    rc, out = sandbox.run_command("pwd")
    assert rc == 0
    assert out.strip() == str(tmp_path.resolve())


def test_timeout_returns_124(tmp_path):
    sandbox = Sandbox(str(tmp_path), SandboxPolicy(mode="yolo", bash_timeout_s=1))
    rc, out = sandbox.run_command("sleep 5")
    assert rc == 124
    assert "timeout" in out


def test_approve_all_requires_confirmer(tmp_path):
    with pytest.raises(SandboxViolation):
        Sandbox(str(tmp_path), SandboxPolicy(mode="approve_all"))


def test_approve_all_decline_refuses(tmp_path):
    sandbox = Sandbox(
        str(tmp_path),
        SandboxPolicy(mode="approve_all"),
        confirm_fn=lambda _: False,
    )
    with pytest.raises(SandboxViolation):
        sandbox.run_command("echo hi")
