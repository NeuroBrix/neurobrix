"""The tagged English text front end: text -> phonemes the way Kokoro's makers' own front end reads it.

kokoro 0.9.4 reads English with misaki 0.9.4 (`G2P(trf=False, fallback=EspeakFallback, unk='')`), which runs spaCy's
en_core_web_sm 3.8.0 (its rule tokenizer, then its part-of-speech tagger) before a lexicon pass. A container that carries
that front end declares it twice, and both declarations are its own data:

* a component whose topology entry has `role: text_front_end` (Kokoro-82M: `text_tagger`) — the tagger network, run by
  the engine's own executor like any component (a graph and its weights; input int64 [n, attributes, keys] of embedding
  rows, output [n, labels] scores);
* `modules/g2p/`: `front_end.json` (kind `lexicon_tagged`: misaki's tables, each with its source; the voice letter's
  variant; `unk`), `tokenizer.json` (spaCy's tokenizer rules), `tagger_meta.json` (the rows' attributes, seeds, row
  counts, the labels, the component's name), `norms.json` (the NORM feature's tables), the variant's gold and silver
  lexicons, and `en_lexicon.txt.gz` (espeak-ng's phonemes of a word list, read for the words no lexicon holds).

A container that declares the front end and lacks any of these is refused by name. A container that declares no
`text_front_end` component keeps the espeak-distilled lexicon runner (`core/module/audio/g2p.py`): the container's own
declaration decides, never the engine.

Here is the text work only — standard library — : preprocess, tokenizer, the tagger's input rows, the lexicon pass. The
network runs where the caller runs it (`tag_scores`, a callable rows -> scores the engine gives: the selected engine,
the request's device); its scores' argmax names each token's tag.
"""
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from neurobrix.core.module.audio.front_end.espeak_list import EspeakList, read_word_list
from neurobrix.core.module.audio.front_end.features import Features
from neurobrix.core.module.audio.front_end.lexicon_pass import FrontEnd, Tables, phonemize
from neurobrix.core.module.audio.front_end.tokenizer import Tokenizer

#: the role a topology component carries when it is the container's text front end
ROLE = "text_front_end"
#: the front end this module reads (front_end.json `kind`)
KIND = "lexicon_tagged"
#: the word list of espeak-ng's phonemes the unknown words are read from
WORD_LIST = "en_lexicon.txt.gz"
_DATA = ("front_end.json", "tokenizer.json", "tagger_meta.json", "norms.json")
_log = logging.getLogger(__name__)


def declared_component(topology: Dict) -> Optional[str]:
    """The component a topology declares as its text front end, or None. Two declare it: refused by name."""
    found = [name for name, info in (topology.get("components") or {}).items()
             if isinstance(info, dict) and info.get("role") == ROLE]
    if len(found) > 1:
        raise ValueError(f"ZERO FALLBACK: the container declares {len(found)} components with role {ROLE!r} "
                         f"({', '.join(found)}); a text has one front end")
    return found[0] if found else None


def _read_json(path: Path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TaggedFrontEnd:
    """One container's tagged front end: its data loaded once, one lexicon pass per variant on first use."""

    def __init__(self, nbx_path: str, component: str, phoneme_lang: str):
        g2p = Path(nbx_path) / "modules" / "g2p"
        missing = [n for n in _DATA + (WORD_LIST,) if not (g2p / n).is_file()]
        if missing:
            raise RuntimeError(
                f"ZERO FALLBACK: this container declares the text front end '{component}' (role {ROLE!r}) but "
                f"{g2p} lacks {', '.join(missing)}: the front end cannot read a text without them. Rebuild the "
                f"container with its text front end data.")
        self.g2p_dir = g2p
        self.component = component
        doc = _read_json(g2p / "front_end.json")
        if doc.get("kind") != KIND:
            raise RuntimeError(f"ZERO FALLBACK: {g2p / 'front_end.json'}: kind {doc.get('kind')!r}; this engine reads "
                               f"the text front end {KIND!r}")
        self.doc = doc
        self.tables = Tables(doc)
        meta = _read_json(g2p / "tagger_meta.json")
        if meta.get("component") != component:
            raise RuntimeError(f"ZERO FALLBACK: {g2p / 'tagger_meta.json'} names the component "
                               f"{meta.get('component')!r}; the topology declares '{component}' as the text front end")
        self.labels: List[str] = list(meta["labels"])
        self.input_name: str = meta["input"]
        self.features = Features(meta, _read_json(g2p / "norms.json"))
        self.tokenizer = Tokenizer(_read_json(g2p / "tokenizer.json"))
        self.variants: Dict[str, str] = doc["voice_variants"]
        self.unk = doc["unk"]
        # the word list speaks the variant of the container's phoneme language: the other variants read no list
        self.list_variant = self.variants.get((phoneme_lang or "")[:1].lower())
        if self.list_variant is None:
            raise RuntimeError(f"ZERO FALLBACK: the container's phoneme_lang {phoneme_lang!r} is none of the voice "
                               f"letters front_end.json gives a variant ({', '.join(sorted(self.variants))})")
        self._words = None
        self._passes: Dict[str, FrontEnd] = {}

    def variant_of(self, voice_letter: str) -> str:
        v = self.variants.get((voice_letter or "")[:1].lower())
        if v is None:
            raise RuntimeError(f"ZERO FALLBACK: a voice '{voice_letter}' has no variant in this container's text front "
                               f"end (front_end.json voice_variants: {self.variants}); ask for a voice whose first "
                               f"letter is one of {', '.join(sorted(self.variants))}")
        return v

    def lexicon_pass(self, variant: str) -> FrontEnd:
        fe = self._passes.get(variant)
        if fe is None:
            fallback = None
            if variant == self.list_variant:
                if self._words is None:
                    self._words = read_word_list(self.g2p_dir / WORD_LIST)
                fallback = EspeakList(self._words, self.tables, variant)
            try:
                fe = FrontEnd(self.doc, str(self.g2p_dir), british=(variant == "gb"), fallback=fallback, unk=self.unk)
            except FileNotFoundError as e:
                raise RuntimeError(f"ZERO FALLBACK: {e}: the '{variant}' voices of this container cannot be read")
            self._passes[variant] = fe
        return fe

    def tokens(self, text: str, variant: str):
        """(preprocessed text, link words, link features, tokenizer tokens) of a text."""
        text2, words, feats = self.lexicon_pass(variant).preprocess(text)
        return text2, words, feats, self.tokenizer(text2)

    def tags(self, scores, n: int) -> List[str]:
        """Each token's tag: the argmax of its row of the tagger's scores ([n, labels])."""
        shape = tuple(getattr(scores, "shape", ()))
        if shape != (n, len(self.labels)):
            raise RuntimeError(f"ZERO FALLBACK: the text front end '{self.component}' answered scores of shape {shape} "
                               f"for {n} tokens; tagger_meta.json declares {len(self.labels)} labels")
        return [self.labels[i] for i in scores.argmax(axis=1).tolist()]

    def phonemes(self, text: str, voice_letter: str,
                 tag_scores: Callable[[List], object]) -> Tuple[str, List]:
        """(phoneme string, the pass's tokens) of a text; `tag_scores(rows)` runs the tagger on the rows."""
        variant = self.variant_of(voice_letter)
        text2, words, feats, toks = self.tokens(text, variant)
        tags = self.tags(tag_scores(self.features.rows_of(toks)), len(toks)) if toks else []
        tagged = [{"text": t["text"], "tag": g, "whitespace": t["whitespace"]} for t, g in zip(toks, tags)]
        phonemes, out = phonemize(tagged, self.lexicon_pass(variant), words, feats)
        _log.debug(f"   [Text front end] {self.component} ({variant}): tags {json.dumps(tags)} phonemes "
                   f"{json.dumps(phonemes, ensure_ascii=False)}")
        return phonemes, out


_LOADED: Dict[Tuple[str, str, str], TaggedFrontEnd] = {}


def declares_front_end(nbx_path_str: str) -> Optional[str]:
    """The text front end component the container's topology declares, or None (no topology, or none declared)."""
    topo = Path(nbx_path_str) / "topology.json"
    return declared_component(_read_json(topo)) if topo.is_file() else None


def for_container(nbx_path_str: str, phoneme_lang: str) -> Optional[TaggedFrontEnd]:
    """The container's tagged front end (loaded once per process), or None when its topology declares none."""
    component = declares_front_end(nbx_path_str)
    if component is None:
        return None
    key = (str(nbx_path_str), component, phoneme_lang)
    fe = _LOADED.get(key)
    if fe is None:
        fe = _LOADED[key] = TaggedFrontEnd(nbx_path_str, component, phoneme_lang)
    return fe
