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
# (needs a device, code): an example after `<!-- needs-gpu -->` runs where a GPU runtime opens, and is skipped where none does
BLOCKS = [(bool(g), c) for g, c in re.findall(r"(<!-- needs-gpu -->\s*)?```python\n(.*?)```", README.read_text(), re.S)]


def _gpu() -> bool:
    try:
        from nbxtensor.memory.allocator import DeviceAllocator
        return DeviceAllocator.device_count() >= 1
    except Exception:
        return False


def test_the_readme_has_examples():
    assert BLOCKS


@pytest.mark.parametrize("i", range(len(BLOCKS)))
def test_every_example_runs(i, tmp_path):
    needs_gpu, code = BLOCKS[i]
    if needs_gpu and not _gpu():
        pytest.skip("README example that launches a kernel: no GPU runtime here")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"README.md example {i + 1}:\n{r.stderr[-4000:]}"

