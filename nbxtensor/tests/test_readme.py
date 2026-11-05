"""Every ```python example of nbxtensor/README.md runs (R-NBXT 10: the README cannot lie).

Each example runs in its own interpreter, as a reader's script would: run inside the suite's process, the settings example's
`set_store(None)` reset the store the engine had set and every later test of that process read an empty certified directory
(the Mac's RUN 41a, 2026-11-05: test_the_live_directory_s_proofs_are_all_labelled red in the suite, green alone)."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)


def test_the_readme_has_examples():
    assert BLOCKS


@pytest.mark.parametrize("i", range(len(BLOCKS)))
def test_every_example_runs(i, tmp_path):
    r = subprocess.run([sys.executable, "-c", BLOCKS[i]], cwd=tmp_path, env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"README.md example {i + 1}:\n{r.stderr[-4000:]}"

