"""The batteries' arms run under the python the launch pinned, never a machine path.

`precision_zoo_campaign.zoo_python()` preferred the rig's `ml/venv` whenever the path existed.
After the engine moved to its own stack, the regression matrix launched under the engine python
ran its first cell under the old venv (2026-09-26, torch 2.5.1). What these tests would do on
the old code on the rig: the first fails — `zoo_python()` answers the legacy path while
`NEUROBRIX_PYTHON` names another interpreter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import precision_zoo_campaign as C  # noqa: E402


def test_the_pinned_engine_python_wins(monkeypatch, tmp_path):
    pinned = str(tmp_path / "engine" / "bin" / "python")
    monkeypatch.delenv("NBX_ZOO_PYTHON", raising=False)
    monkeypatch.delenv("NBX_ZOO_NEUROBRIX", raising=False)
    monkeypatch.setenv("NEUROBRIX_PYTHON", pinned)
    assert C.zoo_python() == pinned
    assert C.nbx_cmd() == [pinned, "-m", "neurobrix"]


def test_without_a_pin_it_is_the_interpreter_running_the_tool(monkeypatch):
    for name in ("NBX_ZOO_PYTHON", "NBX_ZOO_NEUROBRIX", "NEUROBRIX_PYTHON", "NBX_PYTHON"):
        monkeypatch.delenv(name, raising=False)
    assert C.zoo_python() == sys.executable


def test_no_machine_path_is_named():
    src = (Path(C.__file__)).read_text()
    assert "/home/mlops/ml/venv" not in src
