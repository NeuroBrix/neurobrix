"""tools/make_the_proof_table.py, card-less: its command line, its models file, its visible cards, its fresh homes.

The tool runs each model's first launch with a spare card of the class in sight (register row 444: deepseek-moe-16b
on two full V100 32 GB, its proof's worker without room); none of that runs here. What is read before any run is
tested: the arguments, the list, the card set the run is given, the home it is given, the draft path its proofs go to,
and the summary it reads from a run's log.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import make_the_proof_table as T

REPO = Path(__file__).resolve().parents[3]


def test_the_command_line_reads_the_models_the_draft_the_proof_card_and_the_cards(tmp_path):
    """The positional pair, the proof card, the cards, and the homes beside the draft by default."""
    a = T.parse_args([str(tmp_path / "m.txt"), str(tmp_path / "draft"), "--proof-card", "3", "--cards", "2"])
    assert a.proof_card == 3 and a.card_list == ["2"] and a.cards == "2"
    assert a.homes == tmp_path / "draft.homes" and a.hardware is None
    with pytest.raises(SystemExit):
        T.parse_args([str(tmp_path / "m.txt"), str(tmp_path / "draft")])          # --proof-card is required


def test_the_proof_card_among_the_model_s_cards_is_refused_by_name():
    """A proof card the model runs on leaves the proofs no spare card; a card twice or a non-index is refused."""
    with pytest.raises(T.TableRefused, match="holds the proof card 3"):
        T.parse_cards("2,3", 3)
    with pytest.raises(T.TableRefused, match="twice"):
        T.parse_cards("1,1", 3)
    with pytest.raises(T.TableRefused, match="board indices"):
        T.parse_cards("GPU-x", 3)
    assert T.parse_cards(None, 3) is None


def test_the_run_sees_the_model_s_cards_first_and_the_proof_card_last():
    """CUDA_VISIBLE_DEVICES of a run: the cards, then the proof card; without --cards, the given set, which must hold
    the proof card."""
    assert T.visible_devices(["0", "1"], 3, "0,1,2,3") == "0,1,3"
    assert T.visible_devices(None, 3, "2,3") == "2,3"
    with pytest.raises(T.TableRefused, match="does not name the proof card 3"):
        T.visible_devices(None, 3, "0,1")
    with pytest.raises(T.TableRefused, match="does not name the proof card"):
        T.visible_devices(None, 3, None)


def test_the_models_file_is_read_in_order_with_its_requests(tmp_path):
    """A model alone (the census's own request, None here), a model with flags in shell syntax (a `#` inside quotes is
    the request's), comments and blank lines skipped."""
    f = tmp_path / "models.txt"
    f.write_text("# the list\n\nTinyLlama-1.1B-Chat-v1.0\n"
                 "deepseek-moe-16b-chat --prompt \"Say #1, briefly\" --max-tokens 8   # a comment\n")
    assert T.read_models(f) == [("TinyLlama-1.1B-Chat-v1.0", None),
                                ("deepseek-moe-16b-chat", ["--prompt", "Say #1, briefly", "--max-tokens", "8"])]


def test_a_bad_models_file_is_refused_by_name_with_its_line(tmp_path):
    """Twice the same model, a line starting with a flag, an unclosed quote, an empty list: each refused, the line
    named."""
    f = tmp_path / "models.txt"
    for text, why in (("A\nA\n", ":2: A is listed twice"), ("--prompt x\n", ":1: a line starts"),
                      ("A --prompt \"x\n", ":1:"), ("# nothing\n", "no model listed")):
        f.write_text(text)
        with pytest.raises(T.TableRefused, match=why):
            T.read_models(f)


def test_the_example_models_file_lists_the_census_models():
    """The example file reads, and lists exactly the models of the certified census of both volta classes."""
    listed = [m for m, _flags in T.read_models(REPO / "tools" / "proof_table_models.example.txt")]
    census = REPO / "src" / "neurobrix" / "config" / "census" / "nvidia" / "volta"
    for table in ("16g.jsonl", "32g.jsonl"):
        with open(census / table) as fh:
            assert sorted(listed) == sorted({json.loads(line)["model"] for line in fh})


def test_a_home_is_fresh_and_holds_every_place_the_run_writes(tmp_path):
    """The home, the replay cache, Triton's cache and the temporary directory all under one new directory; the run's
    environment carries them, the visible cards, the catalogue and this tree's engine first on the path."""
    home = T.fresh_home(tmp_path / "homes", "TinyLlama-1.1B-Chat-v1.0")
    root = home["root"]
    assert root.parent == tmp_path / "homes" and root.name.startswith("TinyLlama-1.1B-Chat-v1.0-")
    for key in ("NEUROBRIX_HOME", "NEUROBRIX_REPLAY_CACHE", "TRITON_CACHE_DIR", "TMPDIR"):
        assert Path(home[key]).is_dir() and Path(home[key]).is_relative_to(root)
    env = T.run_env(home, "2,3", "/catalogue", base={"PYTHONPATH": "/lib/src", "KEEP": "1"})
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3" and env["NEUROBRIX_CACHE"] == "/catalogue" and env["KEEP"] == "1"
    assert env["PYTHONPATH"].split(":") == [str(REPO / "src"), "/lib/src"] and "root" not in env


def test_the_kept_proofs_go_to_their_class_s_draft_table(tmp_path):
    """The destination is read from the kept file's stamps; kept proofs of two classes are refused by name."""
    kept = tmp_path / "ew_tile_proof.json"
    st = {"vendor": "nvidia", "profile": "volta"}
    kept.write_text(json.dumps({"stamps": {"k1": {"stamp": st, "variants": {}}}}))
    assert T.kept_destination(kept, tmp_path / "d") == tmp_path / "d" / "nvidia" / "volta" / "ew_tile_proof.json"
    kept.write_text(json.dumps({"stamps": {"k1": {"stamp": st}, "k2": {"stamp": {"vendor": "nvidia",
                                                                                  "profile": "ampere"}}}}))
    with pytest.raises(T.TableRefused, match="2 device class"):
        T.kept_destination(kept, tmp_path / "d")


def test_a_run_s_line_reads_its_proved_count_and_its_refusals():
    """The `this run proved N` line's N, and the count of `NOT proven` lines; None when no proof line was printed."""
    log = ("[ew-proof] k: NOT proven — ZERO FALLBACK ...\n"
           "[ew-proof] this run proved 12 element-wise variant(s) in 3.4 s with 2 child process(es); largest ...\n")
    assert T.summary_of(log) == (12, 1)
    assert T.summary_of("nothing\n") == (None, 0)
