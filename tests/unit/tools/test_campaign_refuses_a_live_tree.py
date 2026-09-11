"""A campaign refuses to start unless it measures a frozen tree.

Written after paying for it. On 2026-09-10 a campaign was launched WITHOUT
`--src`, which I read as "the tree measured is the trunk". It is not: the engine
is installed editable and imports `/home/mlops/NeuroBrix_System/src/neurobrix/`,
so without `--src` a campaign measures the **live, mutable** repository. Three
branch merges landed six minutes into its second cell, and every later cell
would have measured a different tree from the first. Fourteen cells of card time
were stopped rather than spent.

The rule was already written — `docs/reference/workshop-layout.md`, that same
afternoon: *a battery runs from a FROZEN worktree*. A written rule was not
enough. This is the door.

Three things are refused at entry, before a second of card is spent:

  * no `--src` at all — the campaign would measure whatever the repository
    happens to be at each moment;
  * a `--src` inside the live repository — the same tree under another name;
  * a `--src` whose worktree is missing the ignored pointers a measurement
    depends on. A fresh checkout has no `.nbx_registry` and no `forge/`, and
    Prism's registry flags then read False there — two GPU hours were lost to
    exactly that on 2026-09-05, and the answer was the change, not the tree.

Run: PYTHONPATH=src python -m pytest tests/unit/tools/test_campaign_refuses_a_live_tree.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[3] / "tools" / "precision_zoo_campaign.py"
REPO = Path(__file__).resolve().parents[3]


def _tool():
    spec = importlib.util.spec_from_file_location("precision_zoo_campaign", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _frozen(tmp_path: Path, *, pointers=("`.nbx_registry`", "forge")) -> Path:
    """A stand-in frozen worktree: a src/ and the ignored pointers beside it."""
    wt = tmp_path / "worktrees" / "frozen_abc1234"
    (wt / "src" / "neurobrix").mkdir(parents=True)
    (wt / "tests").mkdir(parents=True)
    (wt / "tests" / "__init__.py").write_text("")
    if ".nbx_registry" in str(pointers) or "registry" in str(pointers):
        (wt / ".nbx_registry").write_text("{}")
    if "forge" in str(pointers):
        (wt / "forge").mkdir()
    return wt / "src"


def test_no_src_is_refused():
    m = _tool()
    reason = m.frozen_src_refusal(None, REPO)
    assert reason is not None, "a campaign with no --src measures the live tree"
    assert "--src" in reason


def test_the_live_repository_is_refused():
    """The exact shape of the 2026-09-10 loss: --src pointing at the tree the
    author is editing."""
    m = _tool()
    reason = m.frozen_src_refusal(REPO / "src", REPO)
    assert reason is not None
    assert "live" in reason.lower() or "vivant" in reason.lower()


def test_a_path_inside_the_live_repository_is_refused():
    m = _tool()
    assert m.frozen_src_refusal(REPO / "src" / "neurobrix", REPO) is not None


def test_a_frozen_worktree_carrying_its_pointers_is_accepted(tmp_path):
    m = _tool()
    assert m.frozen_src_refusal(_frozen(tmp_path), REPO) is None


def test_a_worktree_missing_the_ignored_pointers_is_refused(tmp_path):
    """A fresh checkout has no .nbx_registry: Prism's flags read False there and
    the measurement blames the change instead of the tree."""
    m = _tool()
    src = _frozen(tmp_path, pointers=("forge",))
    reason = m.frozen_src_refusal(src, REPO)
    assert reason is not None
    assert "nbx_registry" in reason


def test_a_src_that_does_not_exist_is_refused(tmp_path):
    m = _tool()
    assert m.frozen_src_refusal(tmp_path / "nowhere" / "src", REPO) is not None


def test_a_tree_gate_names_a_frozen_worktree_per_arm_and_is_admitted(tmp_path):
    """`--trees` satisfies the rule by construction and must not be refused.

    It WAS refused on 2026-09-11, by the door written the day before: the door
    knew exactly one way of naming a frozen tree. The cost of a door that
    refuses correct usage is that the next person adds `--src` beside `--trees`
    to get past it — which is the shape of every bypass this rule prevents.
    """
    from precision_zoo_campaign import frozen_trees_refusal

    repo = tmp_path / "repo"
    repo.mkdir()
    arms = []
    for label in ("before", "after"):
        w = tmp_path / f"wt_{label}"
        (w / "src").mkdir(parents=True)
        for ptr in (".nbx_registry", "forge"):
            (w / ptr).write_text("")
        arms.append(f"{label}={w / 'src'}")
    assert frozen_trees_refusal(",".join(arms), repo) is None


def test_a_tree_gate_pointing_into_the_live_repo_is_refused_by_arm(tmp_path):
    """Seen failing: the refusal must name WHICH arm, or it cannot be acted on."""
    from precision_zoo_campaign import frozen_trees_refusal

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    good = tmp_path / "wt_ok"
    (good / "src").mkdir(parents=True)
    for ptr in (".nbx_registry", "forge"):
        (good / ptr).write_text("")

    why = frozen_trees_refusal(f"before={repo / 'src'},after={good / 'src'}", repo)
    assert why and "before" in why and "inside the live repository" in why


def test_a_tree_arm_without_a_label_is_refused(tmp_path):
    from precision_zoo_campaign import frozen_trees_refusal
    assert "cannot be named in the verdict" in (
        frozen_trees_refusal("/some/path", tmp_path) or "")
