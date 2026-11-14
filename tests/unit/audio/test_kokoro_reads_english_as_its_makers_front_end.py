"""Kokoro reads English the way its makers' own front end does (kokoro 0.9.4 -> misaki 0.9.4 -> spaCy en_core_web_sm 3.8.0).

The data the vendor's front end gave on the judge texts, the 175 branch sentences of the lexicon pass and a slice of the
documentation's prose (us and gb), recorded once with the vendor's own python (`data/kokoro_front_end_vendor.json.gz`,
its `source` says how), is what the engine's front end (`core/module/audio/front_end`) must give, card-less:

1. the tokenizer's tokens are spaCy's, text and whitespace, on every text;
2. the tagger's input rows are thinc's, on every token of the judge texts and the branch sentences;
3. given the vendor's tags, the phonemes are misaki's byte for byte, with misaki's own fallback outputs handed back for
   the words no lexicon holds (the lexicon pass alone);
4. the espeak word list read through misaki's E2M map gives misaki's EspeakFallback output on a tied list, and on the
   container's untied list everywhere but the words recorded below (espeak's tie lost: schwa then l, i then schwa);
5. a container that declares the front end and lacks one of its files is refused by name; a container that declares
   none keeps the espeak-distilled runner (g2p.py) — the container's own declaration decides.

The data of 1-4 is the container's (`modules/g2p/`): read from the engine's configured model cache, skipped by name when
the container installed here does not carry the front end. The tagger network itself runs on a card:
`tests/regression/test_kokoro_front_end_on_a_card.py`."""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest

from neurobrix.core.module.audio import front_end as FE
from neurobrix.core.module.audio.front_end.espeak_list import EspeakList
from neurobrix.core.module.audio.front_end.lexicon_pass import phonemize

FIXTURE = Path(__file__).parent / "data" / "kokoro_front_end_vendor.json.gz"

#: the untied list's loss, measured over 30 000 list words against misaki on the live espeak: 2.97 % differ, the
#: schwa+l rule about two thirds of them; untying every tied key is the best reading (keeping any one key tied loses
#: 371 to 3 679 more words). The test's bound on its 400-word sample:
UNTIED_LOSS_BOUND = 0.06


def _vendor():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
        return json.load(f)


def _container() -> Path:
    from neurobrix.core.paths import cache_dir
    root = cache_dir() / "Kokoro-82M"
    if not (root / "modules" / "g2p" / "front_end.json").is_file():
        pytest.skip(f"the Kokoro-82M installed in {cache_dir()} does not carry the tagged text front end")
    return root


@pytest.fixture(scope="module")
def front_end():
    root = _container()
    defaults = json.loads((root / "runtime" / "defaults.json").read_text())
    fe = FE.for_container(str(root), defaults["phoneme_lang"])
    assert fe is not None, "the container carries front_end.json but its topology declares no text front end"
    return fe


@pytest.fixture(scope="module")
def vendor():
    return _vendor()


def test_the_tokens_are_spacys(front_end, vendor):
    bad = []
    for row in vendor["texts"]:
        _t2, _w, _f, toks = front_end.tokens(row["text"], row["variant"])
        if [[t["text"], t["whitespace"]] for t in toks] != [[t, w] for t, w, _g in row["tokens"]]:
            bad.append(row["text"][:60])
    assert not bad, f"{len(bad)} texts tokenized otherwise than spaCy: {bad[:5]}"


def test_the_tagger_rows_are_thincs(front_end, vendor):
    n = 0
    for row in vendor["texts"]:
        if "rows" not in row:
            continue
        _t2, _w, _f, toks = front_end.tokens(row["text"], row["variant"])
        assert front_end.features.rows_of(toks) == row["rows"], f"rows differ on {row['text'][:60]!r}"
        n += len(toks)
    assert n > 1000, f"only {n} tokens compared"


def test_the_lexicon_pass_gives_misakis_phonemes(front_end, vendor):
    bad, n = [], 0
    for row in vendor["texts"]:
        recorded = row["fallback"]
        fe = front_end.lexicon_pass(row["variant"])
        replay_of = fe.fallback

        def replay(text, recorded=recorded):
            assert text in recorded, f"the pass asked the fallback {text!r}, misaki never did"
            ps, rating = recorded[text]
            return ps, rating
        fe.fallback = replay
        try:
            text2, words, feats = fe.preprocess(row["text"])
            toks = [{"text": t, "whitespace": w, "tag": g} for t, w, g in row["tokens"]]
            ps, _ = phonemize(toks, fe, words, feats)
        finally:
            fe.fallback = replay_of
        n += 1
        if ps != row["phonemes"]:
            bad.append((row["variant"], row["text"][:60], row["phonemes"][:80], ps[:80]))
    assert n >= 500 and not bad, f"{len(bad)}/{n} differ from misaki: {bad[:3]}"


def test_the_espeak_list_reads_as_misakis_fallback(front_end, vendor):
    tables = front_end.tables
    sample = vendor["espeak_list"]
    tied = EspeakList({w: t for w, t, _ in sample}, tables, "us")
    assert tied.tied
    assert [w for w, _t, m in sample if tied(w)[0] != m] == []
    untied = EspeakList({w: t.replace("^", "") for w, t, _ in sample}, tables, "us")
    assert not untied.tied
    lost = sorted(w for w, _t, m in sample if untied(w)[0] != m)
    assert len(lost) <= UNTIED_LOSS_BOUND * len(sample), f"the untied list loses {len(lost)}/{len(sample)} words: {lost[:10]}"
    tied_keys = [k for k, _ in list(tables.e2m) + list(tables.e2m_variant["us"]) if "^" in k]
    for w, t, _m in sample:
        if w in lost:          # every loss: two characters a tied key spells that espeak did NOT tie there
            assert any(k.replace("^", "") in t.replace(k, "") for k in tied_keys), \
                f"{w!r}: lost though espeak's output {t!r} holds no untied spelling of a tied key"
    assert untied("no such word in the list") == (None, None)


def test_a_container_without_the_declaration_keeps_the_lexicon_runner(tmp_path, monkeypatch):
    from neurobrix.core.module.audio import g2p as G
    (tmp_path / "topology.json").write_text(json.dumps({"components": {"bert": {"type": "neural_component"}}}))
    assert FE.for_container(str(tmp_path), "a") is None
    monkeypatch.setattr(G, "g2p_phonemes", lambda *a, **k: "hə")
    phonemes, ids = G.phoneme_ids("x", str(tmp_path), {"h": 5, "ə": 7}, "a", "af_heart")
    assert (phonemes, ids) == ("hə", [0, 5, 7, 0])


def test_a_declared_front_end_missing_a_file_is_refused_by_name(tmp_path):
    root = _container()
    (tmp_path / "modules").mkdir()
    shutil.copytree(root / "modules" / "g2p", tmp_path / "modules" / "g2p")
    shutil.copy(root / "topology.json", tmp_path / "topology.json")
    for name in ("tokenizer.json", "en_lexicon.txt.gz"):
        (tmp_path / "modules" / "g2p" / name).rename(tmp_path / name)
        with pytest.raises(RuntimeError, match=name.replace(".", r"\.")):
            FE.TaggedFrontEnd(str(tmp_path), "text_tagger", "a")
        (tmp_path / name).rename(tmp_path / "modules" / "g2p" / name)
    (tmp_path / "modules" / "g2p" / "gb_gold.json.gz").unlink()
    fe = FE.TaggedFrontEnd(str(tmp_path), "text_tagger", "a")
    with pytest.raises(RuntimeError, match=r"gb_gold\.json"):
        fe.lexicon_pass("gb")


def test_a_declared_front_end_never_runs_without_its_tagger(tmp_path):
    from neurobrix.core.module.audio import g2p as G
    root = _container()
    defaults = json.loads((root / "runtime" / "defaults.json").read_text())
    with pytest.raises(RuntimeError, match="text_tagger"):
        G.phoneme_ids("Hello.", str(root), defaults["phoneme_vocab"], defaults["phoneme_lang"], "af_heart")


def test_two_front_ends_declared_are_refused():
    topo = {"components": {"a": {"role": FE.ROLE}, "b": {"role": FE.ROLE}}}
    with pytest.raises(ValueError, match="2 components"):
        FE.declared_component(topo)


def test_the_front_end_imports_without_torch():
    import subprocess
    import sys
    src = str(Path(__file__).resolve().parents[3] / "src")
    out = subprocess.run([sys.executable, "-c", "import sys\nimport neurobrix.core.module.audio.front_end\n"
                          "print('torch' in sys.modules)"], capture_output=True, text=True,
                         env={"PYTHONPATH": src, "PATH": "/usr/bin:/bin"}, timeout=120).stdout.strip()
    assert out == "False", f"torch reached the text front end's import chain: {out!r}"


def test_a_tagged_container_plans_its_text_at_the_build_extent(tmp_path):
    """The plan cannot read a tagged container's text (its tags come from a network Prism has not placed yet):
    `planned_phoneme_count` is None there, and len(phoneme_ids) for the lexicon runner's containers."""
    from neurobrix.core.module.audio import g2p as G
    (tmp_path / "topology.json").write_text(json.dumps({"components": {"text_tagger": {"role": FE.ROLE}}}))
    assert G.planned_phoneme_count("Hello.", str(tmp_path), {"h": 1}, "a", "af_heart") is None
