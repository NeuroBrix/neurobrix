"""Every ```python example of nbxtensor/README.md runs (R-NBXT 10: the README cannot lie)."""
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)


def test_the_readme_has_examples():
    assert BLOCKS


@pytest.mark.parametrize("i", range(len(BLOCKS)))
def test_every_example_runs(i):
    exec(compile(BLOCKS[i], f"README.md example {i + 1}", "exec"), {})
