"""The process-group runner the workshop's tools share (the standard library only).

A command runs in its own process group, and a timeout kills the WHOLE group: the command's children too. It imports
nothing of the engine, so a tool outside this repository can hold it as a byte copy.
"""
import subprocess


def run_group(cmd, env, fh, timeout: int, cwd=None, before_kill=None) -> int:
    """The command in its own process group; a timeout kills the WHOLE group — the
    command's children too. A `drift` that timed out at 7200 s on 2026-09-07 left its
    child `run` alive on GPU3 beside the zoo's next model for ten minutes: only the
    direct child was killed. -9 names a timeout. `before_kill(pgid)` runs while the
    group is still alive — the moment its stack can still be read."""
    import os
    import signal
    # the run's developer lines ([Timings], [Peak], the certified-miss line, error stacks) are what these tools parse:
    # a run's default terminal holds none of them (nbxtensor/terminal.py)
    env = {**(os.environ if env is None else env), "NBX_LOG": "debug"}
    p = subprocess.Popen([str(c) for c in cmd], env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd,
                         start_new_session=True)
    try:
        return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if before_kill is not None:
            before_kill(p.pid)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
        return -9
