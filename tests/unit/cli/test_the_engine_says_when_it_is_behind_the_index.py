"""At start the engine says, on stderr, when the installed neurobrix is behind the package index or its nbxtensor does
not match the engine's own pin, and which command updates it. It warns and never refuses; the network call runs in a
detached child process (never on its path, never waited for), at most once a day, its answer cached
on disk; one switch turns it off.

A fake index stands for PyPI (a function standing for the HTTP GET, or a server on the loopback): no test touches the
network.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

from neurobrix import __version__
from neurobrix.cli import update_check as uc


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.delenv(uc.SWITCH, raising=False)
    monkeypatch.setenv("NEUROBRIX_HOME", str(tmp_path))
    return tmp_path


def _args(**kw):
    """The parsed command line the check reads (`json`, and the `serve` child's `ready_fd` / `ready_file`)."""
    return argparse.Namespace(**{"json": False, **kw})


def _index(answers, calls):
    """A fake index: `answers[package]` is the version it publishes, an exception it raises, or a body it returns."""
    def fetch(package, timeout):
        calls.append(package)
        a = answers[package]
        if isinstance(a, Exception):
            raise a
        return a if isinstance(a, dict) else {"info": {"version": a}}
    return fetch


def _have(monkeypatch, nbxtensor_installed=None):
    """The installed nbxtensor and this neurobrix's pin, read from the environment the test runs in."""
    import re
    from pathlib import Path
    from packaging.requirements import Requirement
    pyproject = (Path(__file__).resolve().parents[3] / "pyproject.toml").read_text()
    pin = Requirement(re.search(r'"(nbxtensor[^"]*)"', pyproject).group(1))     # the engine's pin, as it ships
    installed = nbxtensor_installed or uc.installed_version("nbxtensor") or str(pin.specifier).lstrip("=")
    monkeypatch.setattr(uc, "nbxtensor_requirement", lambda: pin)
    monkeypatch.setattr(uc, "installed_version", lambda name: installed if name == "nbxtensor" else __version__)
    return pin, installed


def test_a_newer_release_on_the_index_is_said_with_the_command_that_updates(home, monkeypatch):
    calls = []
    _have(monkeypatch)
    uc.refresh(_index({"neurobrix": "999.0.0", "nbxtensor": "0.0.1"}, calls))
    lines = uc.notice_lines()
    assert any("999.0.0" in line and __version__ in line and uc.update_command("neurobrix") in line for line in lines)
    assert any(uc.SWITCH in line for line in lines)


@pytest.mark.parametrize("latest", [__version__, "0.0.1"])
def test_the_same_or_an_older_release_on_the_index_is_silence(home, monkeypatch, latest):
    _have(monkeypatch)
    uc.refresh(_index({"neurobrix": latest, "nbxtensor": "0.0.1"}, []))
    assert uc.notice_lines() == []


@pytest.mark.parametrize("answer", [OSError("offline"), TimeoutError("timed out"), {"info": {}}, {"info": {"version": "not a release"}}, ["x"]])
def test_an_index_offline_slow_or_wrong_is_silence(home, monkeypatch, answer):
    _have(monkeypatch)
    uc.refresh(_index({"neurobrix": answer, "nbxtensor": answer}, []))
    assert uc.notice_lines() == []


def test_the_real_fetch_gives_up_at_its_timeout_when_nothing_answers(monkeypatch):
    import socket
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)                                   # accepts the connection, never answers
    monkeypatch.setattr(uc, "index_url", lambda: f"http://127.0.0.1:{srv.getsockname()[1]}")
    t0 = time.monotonic()
    with pytest.raises(OSError):
        uc.fetch_index("neurobrix", timeout=0.5)
    assert time.monotonic() - t0 < 2.0
    srv.close()


def test_a_cache_younger_than_a_day_makes_no_network_call(home, monkeypatch):
    calls = []
    uc.write_cache({"checked_at": time.time() - uc.CHECK_EVERY_S / 2, "latest": {}})
    started = []
    monkeypatch.setattr(uc, "_start_refresh", lambda: started.append(1))
    uc.warn(_args())
    assert started == [] and calls == []
    uc.write_cache({"checked_at": time.time() - uc.CHECK_EVERY_S - 1, "latest": {}})
    uc.warn(_args())
    assert started == [1]


def test_the_check_writes_its_time_before_it_asks_so_a_day_has_one_call(home, monkeypatch):
    seen = []
    monkeypatch.setattr(uc, "_start_refresh", lambda: seen.append(json.loads(uc.cache_path().read_text())["checked_at"]))
    uc.warn(_args())
    assert seen and all(time.time() - t < 60 for t in seen)
    assert not uc.due(uc.read_cache())


def test_the_switch_turns_off_every_call_and_every_line(home, monkeypatch):
    monkeypatch.setenv(uc.SWITCH, "1")
    monkeypatch.setattr(uc, "_start_refresh", lambda: pytest.fail("a network check ran with the switch off"))
    uc.write_cache({"checked_at": time.time(), "latest": {"neurobrix": "999.0.0"}})
    assert uc.warn(_args()) == []


def test_json_output_gets_no_line(home, monkeypatch, capsys):
    _have(monkeypatch)
    monkeypatch.setattr(uc, "_start_refresh", lambda: None)
    uc.write_cache({"checked_at": time.time(), "latest": {"neurobrix": "999.0.0"}})
    assert uc.warn(_args(json=True)) == []
    assert capsys.readouterr() == ("", "")


def test_the_line_goes_to_stderr_never_to_stdout(home, monkeypatch, capsys):
    _have(monkeypatch)
    monkeypatch.setattr(uc, "_start_refresh", lambda: None)
    uc.write_cache({"checked_at": time.time(), "latest": {"neurobrix": "999.0.0"}})
    uc.warn(_args())
    out, err = capsys.readouterr()
    assert out == "" and "999.0.0" in err


def test_an_nbxtensor_that_does_not_match_the_pin_is_said_without_the_network(home, monkeypatch):
    pin, _ = _have(monkeypatch, nbxtensor_installed="0.0.1")
    monkeypatch.setattr(uc, "_start_refresh", lambda: None)
    lines = uc.notice_lines()                       # no cache, no index: the local check alone
    assert any("0.0.1" in line and str(pin.specifier) in line and "nbxtensor" in line for line in lines)


def test_a_newer_nbxtensor_the_pin_allows_is_said_and_one_it_refuses_is_silence(home, monkeypatch):
    pin, installed = _have(monkeypatch)
    uc.write_cache({"checked_at": time.time(), "latest": {"neurobrix": __version__, "nbxtensor": "999.0.0"}})
    assert pin.specifier.contains("999.0.0") == any("999.0.0" in line for line in uc.notice_lines())
    monkeypatch.setattr(uc, "nbxtensor_requirement", lambda: __import__("packaging.requirements").requirements.Requirement("nbxtensor>=0"))
    assert any("999.0.0" in line and uc.update_command("nbxtensor") in line for line in uc.notice_lines())


def test_the_command_names_the_interpreter_the_engine_runs_on(monkeypatch):
    import sys
    monkeypatch.setattr(uc, "_install_kind", lambda: ("pip", None))
    assert uc.update_command("neurobrix").startswith(sys.executable.replace(" ", "\\ ") if " " in sys.executable else sys.executable)
    monkeypatch.setattr(uc, "_install_kind", lambda: ("pipx", None))
    assert uc.update_command("neurobrix") == "pipx upgrade neurobrix"


def test_a_cache_file_that_is_not_json_is_silence_and_a_new_check(home, monkeypatch):
    uc.cache_path().write_text("{not json")
    assert uc.read_cache() == {} and uc.due(uc.read_cache())


# ---- the detached child: real processes against a fake index on the loopback ----

def _serve_index(handler_body):
    """A fake index on the loopback: each accepted connection is handed to `handler_body(conn)` in a thread."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    def loop():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=handler_body, args=(conn,), daemon=True).start()
    threading.Thread(target=loop, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.getsockname()[1]}/pypi"


def _command(home, index, switch_off=False):
    """A command that ends at once: the check, then the process exits. Its pipes are read to the end, like a shell's."""
    env = {**os.environ, "NEUROBRIX_HOME": str(home), "NEUROBRIX_PYPI_INDEX": index,
           "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    env.pop(uc.SWITCH, None)
    if switch_off:
        env[uc.SWITCH] = "1"
    code = "import argparse; from neurobrix.cli import update_check as uc; uc.warn(argparse.Namespace(json=False))"
    t0 = time.monotonic()
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    return time.monotonic() - t0, time.time(), done


def _wait(condition, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end and not condition():
        time.sleep(0.05)
    return condition()


def test_a_command_that_ends_at_once_still_gets_the_answer_stored(home):
    answered = []
    def slow_index(conn):                           # answers 1 s after it is asked: after the command has ended
        conn.recv(65536)
        time.sleep(1.0)
        body = json.dumps({"info": {"version": "999.0.0"}}).encode()
        answered.append(time.time())
        conn.sendall(b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s"
                     % (len(body), body))
        conn.close()
    srv, url = _serve_index(slow_index)
    _, ended_at, done = _command(home, url)
    assert done.returncode == 0, done.stderr
    stored = lambda: uc.read_cache().get("latest", {}).get("neurobrix") == "999.0.0"
    assert _wait(stored, uc.TIMEOUT_S + 5), f"no answer stored; cache: {uc.read_cache()}"
    assert answered and answered[0] > ended_at      # the index answered after the command had exited
    srv.close()


def test_a_due_check_adds_no_wait_to_the_command_when_the_index_does_not_answer(tmp_path):
    srv, url = _serve_index(lambda conn: time.sleep(30))     # accepts, never answers: the worst an index can do
    def median(switch_off):
        times = sorted(_command(tmp_path / f"h{switch_off}{i}", url, switch_off)[0] for i in range(5))
        return times[2]
    off, due = median(True), median(False)
    # a wait on the index would cost TIMEOUT_S (3 s); the bound is a third of it, the rest is process noise
    assert due - off < uc.TIMEOUT_S / 3, f"switch off {off:.3f} s, check due {due:.3f} s"
    srv.close()


def test_one_attempt_a_day_with_the_index_down(home):
    asked = []
    def down(conn):                                 # a broken index: takes the connection, drops it
        asked.append(time.time())
        conn.close()
    srv, url = _serve_index(down)
    first = _command(home, url)[2]
    second = _command(home, url)[2]                 # the same day: starts no child
    assert first.returncode == 0 and second.returncode == 0
    _wait(lambda: len(asked) >= 2, uc.TIMEOUT_S + 5)
    time.sleep(uc.TIMEOUT_S / 3)
    assert len(asked) == 2, f"{len(asked)} asks; one child asks each of the two packages once"
    assert not uc.due(uc.read_cache())
    srv.close()


def test_a_census_process_starts_nothing_and_says_nothing(home, monkeypatch, capsys):
    _have(monkeypatch)
    monkeypatch.setenv("NBX_CENSUS", "1")
    monkeypatch.setattr(uc, "_start_refresh", lambda: pytest.fail("a census process started the check"))
    stale = {"checked_at": time.time() - uc.CHECK_EVERY_S - 1, "latest": {"neurobrix": "999.0.0"}}
    uc.write_cache(stale)
    assert uc.warn(_args()) == []
    assert capsys.readouterr() == ("", "") and uc.read_cache() == stale


def test_the_check_comes_after_the_census_shadow_is_installed(home, monkeypatch):
    from neurobrix import cli
    from neurobrix.kernels import census
    order = []
    monkeypatch.setenv("NBX_CENSUS", "1")
    monkeypatch.setattr(census, "install", lambda **kw: order.append("census"))
    monkeypatch.setattr(uc, "warn", lambda args: order.append("check") or [])
    monkeypatch.setattr(sys, "argv", ["neurobrix", "list"])
    try:
        cli.main()
    except BaseException:                           # the command itself may refuse here; only the order is read
        pass
    assert order[:2] == ["census", "check"]


def test_a_stale_temp_file_of_the_check_is_removed_and_nothing_else(home):
    home.mkdir(exist_ok=True)
    old = time.time() - 3600
    stale, young, other = (home / "update_check.json.dead.tmp", home / "update_check.json.live.tmp", home / "x.tmp")
    for f in (stale, young, other):
        f.write_text("{")
    for f in (stale, other):
        os.utime(f, (old, old))
    uc.write_cache({"checked_at": time.time(), "latest": {}})
    uc.warn(_args())
    assert not stale.exists() and young.exists() and other.exists()


def test_serve_says_the_notice_once_its_launched_child_says_nothing(home, monkeypatch, capsys):
    from neurobrix.cli import create_parser
    _have(monkeypatch)
    monkeypatch.setattr(uc, "_start_refresh", lambda: pytest.fail("the launched serve child started a check"))
    uc.write_cache({"checked_at": time.time(), "latest": {"neurobrix": "999.0.0"}})
    parser = create_parser()
    said = uc.warn(parser.parse_args(["serve", "--model", "m"]))                     # the command the person typed
    child = uc.warn(parser.parse_args(["serve", "--model", "m", "--foreground", "--ready-fd", "5"]))
    windows_child = uc.warn(parser.parse_args(["serve", "--model", "m", "--foreground", "--ready-file", "r"]))
    assert any("999.0.0" in line for line in said) and child == [] and windows_child == []
    assert capsys.readouterr().err.count("999.0.0") == 1
