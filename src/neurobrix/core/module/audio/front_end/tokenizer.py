"""The tagged text front end's tokenizer: spaCy's rule tokenizer over the rules the container carries
(`modules/g2p/tokenizer.json`), standard library only (re).

Algorithm studied from spaCy 3.8.16 spacy/tokenizer.pyx and spacy/matcher/phrasematcher.pyx (MIT licence,
Explosion AI); our own code, proven token for token equal to spaCy's on 40 302 texts. Entry point:

    tok = Tokenizer(json_dict)            # the container's tokenizer.json
    tok(text)                             # -> [{"text", "whitespace", "idx"[, "norm"]}], equal to spaCy's
                                          #    [(t.text, t.whitespace_, t.idx) for t in nlp.tokenizer(text)]

The steps, each a port of the vendor's:
1. whitespace split (`_tokenize_affixes`): runs of whitespace and non-whitespace; ONE ' ' after a non-space run
   becomes that run's last token's trailing space, any other whitespace is tokenized like text.
2. per span: a special case (rule) whole; else strip prefixes / suffixes (`_split_affixes`, the loop stops on a
   special case, a token_match, or a string that no longer shrinks; a special case left after removing one prefix
   or one suffix ends it), then the remainder (`_attach_tokens`): special case, token_match / url_match, or split
   on infixes.
3. the special-case pass over the whole document (`_apply_special_cases`): the rules whose text has an affix, an
   infix or a space are matched as token sequences (by ORTH, every match, PhraseMatcher's trie); the longest win,
   then the leftmost; a match whose text with its inner spaces is a rule is replaced by the rule's tokens.
spaCy's span cache is a pure memo of step 2 for the outputs (it never holds a result that involved a special
case, and the entries it fills while loading rules are keyed by rule texts, where the special case wins first),
so it is not reproduced.
"""
import re

_TERMINAL = object()


class Tokenizer:
    def __init__(self, data):
        if data.get("kind") != "spacy_rule_tokenizer":
            raise ValueError(f"tokenizer: kind {data.get('kind')!r} is not 'spacy_rule_tokenizer'")
        for key in ("prefix_search", "suffix_search", "infix_finditer", "token_match", "url_match", "rules",
                    "faster_heuristics"):
            if key not in data:
                raise ValueError(f"tokenizer: the rules lack {key!r}")
        self.prefix_search = self._compile(data["prefix_search"], "search")
        self.suffix_search = self._compile(data["suffix_search"], "search")
        self.infix_finditer = self._compile(data["infix_finditer"], "finditer")
        self.token_match = self._compile(data["token_match"], "match")
        self.url_match = self._compile(data["url_match"], "match")
        self.faster_heuristics = bool(data["faster_heuristics"])
        self.specials = {}
        self.special_norms = {}          # a special case's NORM per token (the tagger's NORM feature reads it)
        for chunk, toks in data["rules"].items():
            self.special_norms[chunk] = [t.get("NORM") for t in toks]
            orths = [t["ORTH"] for t in toks]
            if "".join(orths) != chunk or not orths:
                raise ValueError(f"tokenizer: rule {chunk!r} does not spell its chunk")
            self.specials[chunk] = orths
        # The special-case matcher: spaCy adds a rule when faster_heuristics is off, or the rule has a prefix,
        # an infix, a suffix or a space; its pattern is the rule's own tokenization without special cases.
        self.trie = {}
        for chunk in sorted(self.specials):
            if (not self.faster_heuristics or self.find_prefix(chunk) or self.find_infix(chunk)
                    or self.find_suffix(chunk) or " " in chunk):
                node = self.trie
                for t in self._tokenize_affixes(chunk, False):
                    node = node.setdefault(t[0], {})
                node[_TERMINAL] = chunk

    @staticmethod
    def _compile(entry, call):
        if entry is None:
            return None
        if entry.get("call") != call:
            raise ValueError(f"tokenizer: pattern called {entry.get('call')!r}, expected {call!r}")
        pat = re.compile(entry["pattern"], entry["flags"])
        if pat.flags != entry["flags"]:
            raise ValueError(f"tokenizer: pattern flags {pat.flags} differ from the exported {entry['flags']}")
        return getattr(pat, call)

    # spaCy's find_prefix / find_suffix / find_infix
    def find_prefix(self, s):
        if self.prefix_search is None:
            return 0
        m = self.prefix_search(s)
        return (m.end() - m.start()) if m is not None else 0

    def find_suffix(self, s):
        if self.suffix_search is None:
            return 0
        m = self.suffix_search(s)
        return (m.end() - m.start()) if m is not None else 0

    def find_infix(self, s):
        if self.infix_finditer is None:
            return []
        return list(self.infix_finditer(s))

    # A token in a document is [text, spacy, idx]; idx is set as spaCy's Doc.push_back sets it.
    @staticmethod
    def _push(doc, text, norm=None):
        idx = 0 if not doc else doc[-1][2] + len(doc[-1][0]) + doc[-1][1]
        doc.append([text, False, idx, norm])

    def _tokenize_affixes(self, string, with_special_cases):
        doc = []
        if not string:
            return doc
        i = 0
        start = 0
        in_ws = string[0].isspace()
        for uc in string:
            if uc.isspace() != in_ws:
                if start < i:
                    self._tokenize(doc, string[start:i], with_special_cases)
                if uc == " ":
                    doc[-1][1] = True
                    start = i + 1
                else:
                    start = i
                in_ws = not in_ws
            i += 1
        if start < i:
            self._tokenize(doc, string[start:], with_special_cases)
            doc[-1][1] = string[-1] == " " and not in_ws
        return doc

    def _tokenize(self, doc, span, with_special_cases):
        if with_special_cases and span in self.specials:
            for t, n in zip(self.specials[span], self.special_norms[span]):
                self._push(doc, t, n)
            return
        prefixes, suffixes = [], []
        span = self._split_affixes(span, prefixes, suffixes, with_special_cases)
        self._attach_tokens(doc, span, prefixes, suffixes, with_special_cases)

    def _split_affixes(self, string, prefixes, suffixes, with_special_cases):
        specials = self.specials
        last_size = 0
        prefix = suffix = minus_pre = minus_suf = None
        while string and len(string) != last_size:
            if self.token_match and self.token_match(string):
                break
            if with_special_cases and string in specials:
                break
            last_size = len(string)
            pre_len = self.find_prefix(string)
            if pre_len != 0:
                prefix = string[:pre_len]
                minus_pre = string[pre_len:]
                if minus_pre and with_special_cases and minus_pre in specials:
                    string = minus_pre
                    prefixes.append(prefix)
                    break
            suf_len = self.find_suffix(string[pre_len:])
            if suf_len != 0:
                suffix = string[-suf_len:]
                minus_suf = string[:-suf_len]
                if minus_suf and with_special_cases and minus_suf in specials:
                    string = minus_suf
                    suffixes.append(suffix)
                    break
            if pre_len and suf_len and (pre_len + suf_len) <= len(string):
                string = string[pre_len:-suf_len]
                prefixes.append(prefix)
                suffixes.append(suffix)
            elif pre_len:
                string = minus_pre
                prefixes.append(prefix)
            elif suf_len:
                string = minus_suf
                suffixes.append(suffix)
        return string

    def _attach_tokens(self, doc, string, prefixes, suffixes, with_special_cases):
        for p in prefixes:
            self._push(doc, p)
        if string:
            if with_special_cases and string in self.specials:
                for t, n in zip(self.specials[string], self.special_norms[string]):
                    self._push(doc, t, n)
            elif (self.token_match and self.token_match(string)) or (self.url_match and self.url_match(string)):
                self._push(doc, string)
            else:
                self._push_infixed(doc, string)
        for s in reversed(suffixes):
            self._push(doc, s)

    def _push_infixed(self, doc, string):
        """The remainder split on its infixes (an infix at the very start is not one)."""
        start = 0
        for m in self.find_infix(string):
            infix_start, infix_end = m.start(), m.end()
            if infix_start == 0:
                continue
            if infix_start != start:
                self._push(doc, string[start:infix_start])
            if infix_start != infix_end:
                self._push(doc, string[infix_start:infix_end])
            start = infix_end
        if string[start:]:
            self._push(doc, string[start:])

    def _find_matches(self, doc):
        """PhraseMatcher.find_matches: every (start, end) whose ORTH sequence is a pattern."""
        matches = []
        n = len(doc)
        for start in range(n):
            node = self.trie.get(doc[start][0])
            end = start + 1
            while node is not None:
                if _TERMINAL in node:
                    matches.append((start, end))
                if end >= n:
                    break
                node = node.get(doc[end][0])
                end += 1
        return matches

    def _apply_special_cases(self, doc):
        matches = self._find_matches(doc)
        if not matches:
            return doc
        # _filter_special_spans: longest first, then leftmost; a span is kept when neither its first nor its last
        # token was covered by a span seen before it (kept or not); kept spans in start order.
        seen = set()
        kept = []
        for s, e in sorted(matches, key=lambda m: (-(m[1] - m[0]), m[0])):
            if s not in seen and (e - 1) not in seen:
                kept.append((s, e))
            seen.update(range(s, e))
        span_data = {}
        for s, e in sorted(kept):
            text = "".join(doc[k][0] + (" " if doc[k][1] else "") for k in range(s, e - 1)) + doc[e - 1][0]
            span_data[s] = (text, e)
        out = []
        i = 0
        while i < len(doc):
            if i not in span_data:
                out.append(doc[i])
                i += 1
                continue
            text, e = span_data[i]
            rule = self.specials.get(text)
            if rule is None:
                out.extend(doc[i:e])
            else:
                orig_final_spacy = doc[e - 1][1]
                idx = doc[i][2]
                for t, n in zip(rule, self.special_norms[text]):
                    out.append([t, False, idx, n])
                    idx += len(t)
                out[-1][1] = orig_final_spacy
            i = e
        return out

    def __call__(self, text):
        doc = self._tokenize_affixes(text, True)
        doc = self._apply_special_cases(doc)
        return [{"text": t[0], "whitespace": " " if t[1] else "", "idx": t[2],
                 **({"norm": t[3]} if len(t) > 3 and t[3] is not None else {})} for t in doc]

