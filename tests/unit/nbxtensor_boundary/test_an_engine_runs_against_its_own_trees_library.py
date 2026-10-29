"""An engine imported from a source tree runs against the nbxtensor of THAT tree, whatever editable install the
environment carries: the install points at one tree, and a process of another tree that loaded it ran its engine against
a different library (the R33 guard on d9fc8747, 2026-10-29: the engine provided a service the other tree's library no
longer declared). A library already imported from elsewhere is refused by name. A second tree is built here from links."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _tree(tmp_path):
    other = tmp_path / "other_tree"
    (other / "nbxtensor").mkdir(parents=True)
    os.symlink(REPO / "src", other / "src")
    os.symlink(REPO / "pyproject.toml", other / "pyproject.toml")
    lib = other / "nbxtensor" / "src"
    lib.mkdir()
    shutil.copytree(REPO / "nbxtensor" / "src" / "nbxtensor", lib / "nbxtensor")   # two trees never share one library
    return other


def _run(other, code):
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(other / "src"), "PYTHONNOUSERSITE": "1",
           "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env,
                          cwd=other)


def test_the_engine_loads_the_library_of_its_tree(tmp_path):
    other = _tree(tmp_path)
    out = _run(other, "import neurobrix, nbxtensor; print(nbxtensor.__path__[0])")
    assert out.returncode == 0, out.stderr[-1500:]
    assert out.stdout.strip() == str(other / "nbxtensor" / "src" / "nbxtensor"), out.stdout


def test_a_library_imported_from_elsewhere_first_is_refused_by_name(tmp_path):
    other = _tree(tmp_path)
    out = _run(other, f"import sys; sys.path.insert(0, {str(REPO / 'nbxtensor' / 'src')!r}); import nbxtensor; import neurobrix")
    assert out.returncode != 0 and "nbxtensor was imported from" in out.stderr, (out.returncode, out.stderr[-1500:])
