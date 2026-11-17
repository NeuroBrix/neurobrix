"""No path this repository tracks names the builder: its tools and tests live in the builder's own repository, and a
public file name is public text. The name is matched in any letter case; "forget" and its family are other words and
pass (tests/unit/kernels/test_a_profile_that_forgets_its_ladder_is_refused.py). .gitignore holds new paths to the same
rule; this test holds the tracked list, which an ignore rule never reaches (`git add -f`, a path tracked before it).

Injection (seen red, then restored green): a scratch path naming the builder added with `git add -f` -> named here.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
#: the builder's name, spelled so that this file does not carry it; not followed by "t" ("forget", "forgets")
BUILDER = re.compile("f" + "orge(?!t)", re.IGNORECASE)


def test_no_tracked_path_names_the_builder():
    if not (ROOT / ".git").exists() or shutil.which("git") is None:
        pytest.skip(f"{ROOT} is not a git checkout (an sdist): no tracked list to read")
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True,
                             text=True).stdout.split("\0")
    named = [p for p in tracked if BUILDER.search(p)]
    assert not named, f"tracked paths name the builder (move them to the builder's repository): {named}"
