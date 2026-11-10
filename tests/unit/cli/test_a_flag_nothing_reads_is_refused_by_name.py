"""A request flag the command line accepts and no flow reads is refused by name before anything loads.

Before (2026-11-10): `neurobrix run --reference-audio voice.wav` was accepted — `cmd_run` stored the path under a key no
flow reads — and the model spoke with its own voice: chatterbox, same sentence and seed, gave the same bytes with and
without the reference. `--reference-image` was not even stored.

What this test does if the code is wrong: a listed flag is accepted (no refusal, or one that does not name the flag),
a request without it is refused, or a flag is in the table although a flow reads it (its key appears in a flow or the
engine: then the table lies and the flag must leave it) — red. Red on main: no refusal exists.

No card. Run: python -m pytest tests/unit/cli/test_a_flag_nothing_reads_is_refused_by_name.py
"""
import re
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"


def test_each_unread_flag_is_refused_naming_it():
    from neurobrix.cli.utils import UNREAD_FLAGS, refuse_unread_flags
    assert set(UNREAD_FLAGS) == {"reference_audio", "reference_image"}
    for field, (flag, _why) in UNREAD_FLAGS.items():
        with pytest.raises(SystemExit) as e:
            refuse_unread_flags(NS(**{field: "x.wav"}))
        assert "ZERO FALLBACK" in str(e.value) and flag in str(e.value)
    refuse_unread_flags(NS(reference_audio=None, reference_image=None))
    refuse_unread_flags(NS())


def test_run_refuses_before_anything_loads(monkeypatch):
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands import run as run_mod
    monkeypatch.setattr(run_mod, "find_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the model was looked up")))
    args = create_parser().parse_args(["run", "--model", "any/model", "--prompt", "hi", "--reference-audio", "voice.wav"])
    with pytest.raises(SystemExit, match="--reference-audio"):
        run_mod.cmd_run(args)


def test_the_table_holds_only_flags_no_flow_reads():
    """The day a flow reads `global.<field>` (or its `_path`), the flag is wired: it must leave the table."""
    from neurobrix.cli.utils import UNREAD_FLAGS
    code = "\n".join(p.read_text() for d in ("core", "triton", "serving") for p in (SRC / d).rglob("*.py"))
    for field in UNREAD_FLAGS:
        assert not re.search(rf"global\.{field}(_path)?\b", code), f"a flow reads global.{field}: remove it from UNREAD_FLAGS"
