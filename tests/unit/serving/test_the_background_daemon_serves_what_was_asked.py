"""`neurobrix serve` without `--foreground` on macOS and Windows starts the
daemon as a child process running `serve --foreground`: that child must serve
the hardware profile, execution mode and idle timeout the user asked for.

Before (found 2026-10-11 while rewriting the ready signal, supervisor queue
item 6): the child command carried only `--model` and `--foreground`, so a
background `serve --triton --hardware X --timeout 60` on an M4 served the
compiled mode on the auto-detected profile with the default 1800 s timeout.

The child's command is read back through the real CLI parser and must say
what the parent was asked.
"""
from __future__ import annotations

import pytest

from neurobrix.cli import create_parser
from neurobrix.serving.server import ServingDaemon

from test_the_daemon_answers_while_it_generates import FakeEngine


class AskedEngine(FakeEngine):
    def __init__(self, hardware_id, mode):
        super().__init__()
        self.hardware_id, self.mode = hardware_id, mode


_MODE_FLAG = {"compiled": "compiled", "sequential": "sequential", "triton": "triton",
              "triton_sequential": "triton_sequential"}


@pytest.mark.parametrize("mode", sorted(_MODE_FLAG))
def test_the_child_command_carries_hardware_mode_and_timeout(mode):
    d = ServingDaemon("fake", "m4-pro-48g", mode=mode, idle_timeout=60.0,
                      engine=AskedEngine("m4-pro-48g", mode))
    cmd = d._foreground_command()
    assert cmd[1:3] == ["-m", "neurobrix"]
    args = create_parser().parse_args(cmd[3:])
    assert args.command == "serve" and args.foreground and args.model == "fake"
    assert args.hardware == "m4-pro-48g"
    assert args.timeout == 60
    assert getattr(args, _MODE_FLAG[mode]) is True
    others = set(_MODE_FLAG.values()) - {_MODE_FLAG[mode]}
    assert not any(getattr(args, o) for o in others)
