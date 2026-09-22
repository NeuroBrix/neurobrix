"""`python -m neurobrix` must exit with the status its command returned.

Red on 2026-09-23: `__main__.py` was

    from neurobrix.cli import main
    main()

so every `return cmd_xxx(args)` in `cli/__init__.py` was discarded and the process exited 0
whatever the command decided. `autotune certify` computes

    return 0 if not bad and not summary["failed"] else 1

with a comment explaining why `unreachable` is deliberately NOT folded into that status — and
none of it could ever be observed. A certification run that FAILED a key exited 0.

Measured: the Apple campaign's addmm family reported `"failed": 1` in its own summary and the
shell read `rc=0`, so the batched runner recorded the family COMPLETE with a census key
uncertified. Any CI step gating on this command got a green that could not go red.

This is the vacuous-gate class in its purest form: not a test that checks too little, but a
status that cannot express failure.
"""
import subprocess
import sys
import textwrap


def _run(code: str):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          capture_output=True, text=True)


def test_a_commands_return_value_becomes_the_exit_status():
    """Patch the command layer to return 3; the process must exit 3."""
    r = _run("""
        import runpy, sys
        import neurobrix.cli as cli
        cli.main = lambda: 3
        sys.argv = ["neurobrix", "autotune", "check"]
        runpy.run_module("neurobrix", run_name="__main__")
    """)
    assert r.returncode == 3, (
        f"a command returned 3 and the process exited {r.returncode}: the return value is "
        f"discarded, so no neurobrix command can report failure by its status.\n{r.stderr[-400:]}"
    )


def test_success_still_exits_zero():
    """A command returning 0 or None must not become a non-zero status."""
    for value in ("0", "None"):
        r = _run(f"""
            import runpy, sys
            import neurobrix.cli as cli
            cli.main = lambda: {value}
            sys.argv = ["neurobrix", "autotune", "check"]
            runpy.run_module("neurobrix", run_name="__main__")
        """)
        assert r.returncode == 0, f"returning {value} exited {r.returncode}"


def test_a_non_integer_return_does_not_become_a_status():
    """A command that returns a string must not exit 1 with that string on stderr."""
    r = _run("""
        import runpy, sys
        import neurobrix.cli as cli
        cli.main = lambda: "done"
        sys.argv = ["neurobrix", "autotune", "check"]
        runpy.run_module("neurobrix", run_name="__main__")
    """)
    assert r.returncode == 0, f"a string return exited {r.returncode}: {r.stderr[-200:]}"
