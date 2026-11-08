"""A command's default terminal holds what its user needs, and nothing else.

The terminal-noise inventory of 0.6.2 (2026-11-08, every line of every command classed useful / developer / debris
with its emitter): a successful `run` printed 2-3 useful lines out of 45-50, an `import` 2 out of 36-38, `hub` and
`list` framed their table in banners and separators, `--explain-plan --json` printed its one record over 120 lines,
and Triton warned about our own kernel. The rule since: stdout carries the result (the text, the transcript, the
`SAVED:` path, the table, the JSON record on ONE line); stderr at most a few status lines (the engine, the time, a
first launch measuring the machine, a warning, an error and its remedy); every developer line is under `-v` /
`NBX_LOG=debug` (nbxtensor/terminal.py), every debris line removed at its source.

Each test launches the command through the entry point as the user types it, counts the lines on each stream and
fails if a developer or debris line of the inventory reappears. Card-bound tests (a model run, serve) run on the one
card the process sees and skip without one; they read the installed models where the user's run would.

What these tests would do if the code were wrong: a banner, a separator, an echo of the request, a `[n/4]` phase, a
plan line, a `[Timings]` / `[Peak]` / `[Replay]` / `[autotune]` line, a traceback after a stated error, the JSON on
several lines, a dev command in `--help`, Triton's non-boolean `tl.where` warning on the first compile of
`masked_fill` — each is a pattern below, and each fails its test (seen red on main caafe74c8, 2026-11-08)."""
from __future__ import annotations

import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PY = sys.executable
ASSET_WAV = REPO / "tests" / "assets" / "test_speech_ref.wav"

#: Developer (D) and debris (X) lines of the inventory, by their text: none may reach a default terminal.
NOISE = [
    r"^=+$", r"^-{5,}$", r"^─{5,}$",                                  # separators
    r"NeuroBrix (Run|Import|Models|Hub|Remove)\b", r"IMPORT COMPLETE",  # banners
    r"^(Model|Hardware|Prompt|Audio|Registry|Search|Cache): ",         # echo of the request, registry URL
    r"^\s*\[\d/4\]", r"^Container: built", r"Components: \[",          # run / import phases
    r"\[Auto-detect\]", r"^\s*Profile: ", r"^\s*Strategy: ", r"^\s*Why: ", r"^\s*Devices: ", r"^\s*Host: ",
    r"^\s*\S+ → \S+$", r"CLI inputs:", r"Total inputs:", r"\[Execute\]", r"^\s*Engine: [A-Z]",
    r"\[(Triton|Compiled)\] '", r"\[Triton\] \d+ constant", r"no calibration record", r"precision contract from",
    r"\[Replay\]", r"\[StepSlab\]", r"\[autotune\] certified directory", r"\[Timings\]", r"\[Peak\]",
    r"\[RuntimeGrowth\]", r"\[Output\]", r"\[Success\]", r"declined —", r"\[Allocator\]", r"global\.seed",
    r"\[Audio", r"\[model\.(encoder|decoder)\]", r"\[plan\] --explain-plan", r"memory budget —",
    r"Traceback \(most recent call last\)", r"non-boolean condition", r"unified memory — planning against",
    r"^\s*(Category|Description|Size|Disk|Vendor|Saved|Extracted|Store): ", r"LICENSE NOTICE",
    r"License accepted via", r"Remove with:", r"Store usage:", r"^Compute environment$",
    r"^(Python|Version|Platform|Scripts dir|neurobrix found): ", r"GPU drivers: +NVIDIA", r"^\s*(PyTorch|built for|NVIDIA driver): ",
    r"\[Serve\] (Starting|Model:|Hardware:|Mode:|Idle timeout|Loading model|Log:)", r"\[Chat\] Disconnected",
    r"\[Stop\] Sending",
]


def _env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("NBX_LOG", "NBX_DEBUG")}
    env.update(PYTHONNOUSERSITE="1",
               PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO / "nbxtensor" / "src")]),
               PATH=os.pathsep.join([str(Path(PY).parent), env.get("PATH", "")]))
    env.update(extra)
    return env


def _neurobrix(*argv, env=None, timeout=900, stdin=None):
    p = subprocess.run([PY, "-m", "neurobrix", *argv], capture_output=True, text=True, env=env or _env(),
                       timeout=timeout, input=stdin, cwd=str(REPO))
    return p, p.stdout.splitlines(), p.stderr.splitlines()


def _no_noise(lines, where):
    found = [(pat, l) for l in lines for pat in NOISE if re.search(pat, l)]
    assert not found, f"{where}: developer or debris lines on the default terminal: {found[:6]}"


def _home(tmp_path) -> Path:
    home = tmp_path / "home"
    (home / "cache").mkdir(parents=True)
    (home / "store").mkdir()
    return home


def _fake_model(home: Path, name: str, family: str = "llm", license_id: str = "apache-2.0") -> None:
    d = home / "cache" / name
    (d / "components").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"model_name": name, "family": family, "license": license_id}))
    (d / "topology.json").write_text(json.dumps({"components": {}}))


class _Registry(http.server.BaseHTTPRequestHandler):
    """The hub's model list, served locally: the table is the command's, the network is not under test."""
    MODELS = [{"slug": f"org{i}/model-with-a-rather-long-name-{i}", "name": f"model-with-a-rather-long-name-{i}",
               "category": "LLM", "fileSize": 2_100_000_000, "license": "apache-2.0", "downloadCount": i}
              for i in range(3)]

    def do_GET(self):  # noqa: N802
        body = json.dumps({"models": self.MODELS, "total": len(self.MODELS)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def registry():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Registry)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


# ── cardless commands ─────────────────────────────────────────────────────────────────────────────────────────────

def test_importing_the_packages_prints_nothing():
    p = subprocess.run([PY, "-c", "import nbxtensor, neurobrix, neurobrix.cli"], capture_output=True, text=True,
                       env=_env(CUDA_VISIBLE_DEVICES=""), timeout=300)
    assert p.returncode == 0, p.stderr
    assert (p.stdout, p.stderr) == ("", "")


def test_help_lists_the_user_commands_only():
    p, out, err = _neurobrix("--help", env=_env(CUDA_VISIBLE_DEVICES=""))
    assert p.returncode == 0 and not err
    text = "\n".join(out)
    for dev in ("calibrate", "autotune", "drift", "coverage"):
        assert not re.search(rf"\b{dev}\b", text), f"--help lists the developer command {dev}"
    assert "c4140-v100x4-nvlink" not in text and "Flex.1-alpha/model.nbx" not in text
    assert "NBX_LOG=debug" in text and "-v, --verbose" in text
    assert len(out) <= 40, len(out)
    _, out_v, _ = _neurobrix("--help", "-v", env=_env(CUDA_VISIBLE_DEVICES=""))
    assert all(re.search(rf"\b{dev}\b", "\n".join(out_v)) for dev in ("calibrate", "autotune", "drift", "coverage"))


def test_hub_is_the_table_and_one_summary_line(registry, tmp_path):
    p, out, err = _neurobrix("hub", "--registry", registry, env=_env(NEUROBRIX_HOME=str(_home(tmp_path)),
                                                                     CUDA_VISIBLE_DEVICES=""))
    assert p.returncode == 0 and not err, err
    _no_noise(out, "hub")
    assert len(out) == 1 + len(_Registry.MODELS) + 1, out          # header, rows, summary
    header, *rows, summary = out
    assert header.split() == ["MODEL", "CATEGORY", "SIZE", "LICENSE", "DL"]   # the empty STATUS column is dropped
    col = header.index("CATEGORY")
    assert all(r[col - 2:col] == "  " and r[col] != " " for r in rows), rows  # columns sized to the longest name
    assert summary.startswith(f"{len(_Registry.MODELS)} model(s)")


def test_hub_verbose_names_the_registry(registry, tmp_path):
    p, out, err = _neurobrix("hub", "--registry", registry, "-v",
                             env=_env(NEUROBRIX_HOME=str(_home(tmp_path)), CUDA_VISIBLE_DEVICES=""))
    assert p.returncode == 0 and f"Registry: {registry}" in err


def test_import_of_an_installed_model_is_one_line(tmp_path):
    home = _home(tmp_path)
    _fake_model(home, "TinyLlama-1.1B-Chat-v1.0")
    p, out, err = _neurobrix("import", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "--accept-license",
                             "--registry", "http://127.0.0.1:9", env=_env(NEUROBRIX_HOME=str(home),
                                                                         CUDA_VISIBLE_DEVICES=""))
    assert p.returncode == 0, err
    assert out == ["TinyLlama-1.1B-Chat-v1.0 is already installed (--force re-downloads)."] and not err


def test_list_is_the_table_and_one_summary_line(tmp_path):
    home = _home(tmp_path)
    for n in ("a-model", "another-model-with-a-long-name"):
        _fake_model(home, n)
    (home / "store" / "a-model.nbx").write_bytes(b"x" * 10)
    p, out, err = _neurobrix("list", env=_env(NEUROBRIX_HOME=str(home), CUDA_VISIBLE_DEVICES=""))
    assert p.returncode == 0 and not err, err
    _no_noise(out, "list")
    assert len(out) == 1 + 2 + 1, out
    assert out[-1].startswith("2 model(s) · ") and "neurobrix clean --store" in out[-1]


def test_chat_and_stop_without_a_daemon_say_one_line(tmp_path):
    env = _env(NEUROBRIX_HOME=str(_home(tmp_path)), CUDA_VISIBLE_DEVICES="")
    p, out, err = _neurobrix("chat", env=env, stdin="")
    assert p.returncode == 1 and not out and len(err) == 1 and err[0].startswith("Error: no daemon"), (out, err)
    p, out, err = _neurobrix("stop", env=env)
    assert p.returncode == 0 and out == ["No daemon is running."] and not err


def test_a_level_that_is_no_level_is_refused_by_name(tmp_path):
    p, out, err = _neurobrix("list", env=_env(NEUROBRIX_HOME=str(_home(tmp_path)), NBX_LOG="loud",
                                              CUDA_VISIBLE_DEVICES=""))
    assert p.returncode == 2 and not out and err == [
        "Error: NBX_LOG='loud' is not a level: use one of debug, info, warning, error"]


def test_masked_fill_gives_triton_a_boolean_condition(tmp_path):
    """The first compile of masked_fill on a uint8 mask (the causal bias's, TinyLlama --triton) made Triton warn
    'tl.where with a non-boolean condition'; compiled cardless for sm_70, it warns no more."""
    code = (
        "import warnings, triton\n"
        "from triton.compiler import ASTSource\n"
        "from triton.backends.compiler import GPUTarget\n"
        "from nbxtensor.ops.masked_fill import masked_fill_forward_kernel as k\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    triton.compile(ASTSource(fn=k, signature={'input_ptr': '*fp16', 'mask_ptr': '*u8', 'output_ptr': '*fp16',"
        " 'n_elements': 'i32', 'fill_value': 'fp32', 'BLOCK_SIZE': 'constexpr'}, constexprs={'BLOCK_SIZE': 1024}),"
        " target=GPUTarget('cuda', 70, 32))\n"
        "print(sum('non-boolean' in str(x.message) for x in w))\n")
    p = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=600,
                       env=_env(CUDA_VISIBLE_DEVICES="", TRITON_CACHE_DIR=str(tmp_path / "triton")))
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.strip() == "0", "masked_fill hands tl.where a uint8 condition"


# ── card-bound commands: the one card this process sees ───────────────────────────────────────────────────────────

def _one_card() -> bool:
    try:
        from nbxtensor.memory.allocator import DeviceAllocator
        return DeviceAllocator.device_count() == 1
    except Exception:
        return False


def _installed(model: str) -> bool:
    from neurobrix.core.paths import cache_dir
    return (cache_dir() / model / "manifest.json").exists()


card = pytest.mark.skipif(not _one_card(), reason="needs exactly one visible card (CUDA_VISIBLE_DEVICES=<n>)")
ENGINES = [pytest.param([], id="default"), pytest.param(["--triton"], id="triton")]


def _status_only(err, where):
    """stderr of a run: a few status lines — the engine, the time, a warning — never a developer line."""
    _no_noise(err, where)
    assert len(err) <= 4, f"{where}: {len(err)} stderr lines: {err}"
    assert any(l.startswith("[Engine] engine: ") for l in err), err
    assert any(l.startswith("[Timing] Total execution: ") for l in err), err


@card
@pytest.mark.parametrize("engine", ENGINES)
def test_an_llm_run_prints_its_text_only(engine):
    model = "TinyLlama-1.1B-Chat-v1.0"
    if not _installed(model):
        pytest.skip(f"{model} is not installed")
    p, out, err = _neurobrix("run", "--model", model, "--prompt", "What is the capital of France?",
                             "--max-tokens", "32", *engine)
    assert p.returncode == 0, err[-20:]
    assert out and "Paris" in out[0] and len(out) <= 2, out
    _no_noise(out, "run llm stdout")
    _status_only(err, "run llm stderr")


@card
@pytest.mark.parametrize("engine", ENGINES)
def test_an_image_run_prints_its_saved_path_only(engine, tmp_path):
    model = "Sana_1600M_1024px_MultiLing"
    if not _installed(model):
        pytest.skip(f"{model} is not installed")
    dest = tmp_path / "apple.png"
    p, out, err = _neurobrix("run", "--model", model, "--prompt", "a red apple on a wooden table", "--steps", "8",
                             "--height", "512", "--width", "512", "--seed", "42", "--output", str(dest), *engine)
    assert p.returncode == 0, err[-20:]
    assert out == [f"SAVED: {dest}"] and dest.exists(), out
    _status_only(err, "run image stderr")


@card
@pytest.mark.parametrize("engine", ENGINES)
def test_a_transcription_prints_its_text_only(engine):
    model = "whisper-large-v3-turbo"
    if not _installed(model):
        pytest.skip(f"{model} is not installed")
    p, out, err = _neurobrix("run", "--model", model, "--audio", str(ASSET_WAV), *engine)
    assert p.returncode == 0, err[-20:]
    assert len(out) == 1 and "slushy country roads" in out[0], out
    _status_only(err, "run stt stderr")


@card
def test_the_plan_as_json_is_one_line_and_nothing_else():
    model = "TinyLlama-1.1B-Chat-v1.0"
    if not _installed(model):
        pytest.skip(f"{model} is not installed")
    p, out, err = _neurobrix("run", "--model", model, "--prompt", "What is the capital of France?",
                             "--max-tokens", "32", "--explain-plan", "--json")
    assert p.returncode == 0, err[-20:]
    assert len(out) == 1, f"{len(out)} stdout lines"
    assert json.loads(out[0])["schema"].startswith("neurobrix.explain-plan/")
    assert not err, err


@card
def test_verbose_restores_the_developer_lines_the_gate_reads():
    model = "TinyLlama-1.1B-Chat-v1.0"
    if not _installed(model):
        pytest.skip(f"{model} is not installed")
    p, out, err = _neurobrix("run", "--model", model, "--prompt", "What is the capital of France?",
                             "--max-tokens", "8", "--triton", env=_env(NBX_LOG="debug"))
    assert p.returncode == 0, err[-20:]
    text = "\n".join(err)
    for line in ("[Timings] load_s=", "[Peak] device_mb=", "[autotune] certified directory:", "Strategy: "):
        assert line in text, f"NBX_LOG=debug lost {line!r}"
    assert out and "Paris" in out[0]


@card
def test_doctor_says_the_engines_and_the_verdict():
    p, out, err = _neurobrix("doctor")
    assert p.returncode == 0 and not err, err
    _no_noise(out, "doctor")
    assert out[0].startswith("neurobrix ") and "on PATH" in out[0]
    assert any(l.strip().startswith("Triton engine:") for l in out)
    assert out[-1].startswith("No blocking problem found")
    assert len(out) <= 6, out


@card
def test_serve_chat_and_stop_say_one_line_each(tmp_path):
    model = "TinyLlama-1.1B-Chat-v1.0"
    if not _installed(model):
        pytest.skip(f"{model} is not installed")
    from neurobrix.core.paths import cache_dir, engine_home
    home = tmp_path / "home"
    (home / "hardware").mkdir(parents=True)
    profiles = sorted((engine_home() / "hardware").glob("default-*.yml"))
    if not profiles:
        pytest.skip("no generated hardware profile to serve with")
    # the profile of the visible card, generated beside the user's engine home (no first-launch measurement here)
    sys.path.insert(0, str(REPO / "src"))
    from neurobrix.core.prism.autodetect import _visible_set_tag
    hw = f"default-{_visible_set_tag()}"
    if not (engine_home() / "hardware" / f"{hw}.yml").exists():
        pytest.skip(f"no profile {hw} for this card")
    shutil.copy(engine_home() / "hardware" / f"{hw}.yml", home / "hardware" / f"{hw}.yml")
    env = _env(NEUROBRIX_HOME=str(home), NEUROBRIX_CACHE=str(cache_dir()))
    try:
        p, out, err = _neurobrix("serve", "--model", model, "--hardware", hw, "--triton", "--timeout", "300", env=env)
        assert p.returncode == 0, (out, err, (home / "daemon.log").read_text()[-3000:]
                                   if (home / "daemon.log").exists() else "")
        assert not out and len(err) <= 2, (out, err)
        assert err[-1].startswith(f"Serving {model}") and "neurobrix chat" in err[-1]
        _no_noise(err, "serve")
        p, out, err = _neurobrix("chat", "--max-tokens", "16", env=env, stdin="What is the capital of France?\n")
        assert p.returncode == 0 and not err, err
        _no_noise(out, "chat")
        assert out[0] == f"{model} — /new /context /status /quit" and len(out) <= 3, out
        assert any(l.startswith("You: Assistant: ") and "Paris" in l for l in out), out
    finally:
        p, out, err = _neurobrix("stop", env=env)
    assert out == ["Stopped the serving daemon."] and not err, (out, err)


# ── the library's own command line ────────────────────────────────────────────────────────────────────────────────

def test_install_metal_downloads_nothing_already_installed(monkeypatch, capsys):
    from nbxtensor.backend import install_metal as M
    monkeypatch.setattr(M.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(M.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(M.platform, "mac_ver", lambda: ("15.4", ("", "", ""), "arm64"))
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    base = M.RELEASES + "apple-v0.1.0%2Btriton3.8.0/"
    t = f"{base}triton-3.8.0%2Bgit4a15f415.nbx1-{tag}-{tag}-macosx_15_0_arm64.whl"
    b = f"{base}nbx_triton_apple-0.1.0-{tag}-{tag}-macosx_15_0_arm64.whl"
    monkeypatch.setattr(M, "METAL_WHEELS", {tag: ((t, "a" * 64), (b, "b" * 64))})
    installed = {"triton": "3.8.0+git4a15f415.nbx1", "nbx-triton-apple": "0.1.0"}
    monkeypatch.setattr(M, "_installed_version", lambda dist: installed.get(dist))
    calls = []
    monkeypatch.setattr(M.subprocess, "call", lambda *a, **k: calls.append(a) or 0)
    assert M.main("0.0.5") == 0 and not calls, "a download although the pinned wheels are installed"
    assert capsys.readouterr().out.splitlines() == [
        "Triton for Metal already installed (triton 3.8.0+git4a15f415.nbx1, nbx-triton-apple 0.1.0)."]
    installed["triton"] = "3.8.0"                       # PyPI's Triton, not the Metal build: installed over
    assert M.main("0.0.5") == 0 and len(calls) == 1
    assert capsys.readouterr().out.startswith("Installing Triton for Metal")
