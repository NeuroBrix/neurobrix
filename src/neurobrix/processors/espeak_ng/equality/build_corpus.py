"""Builds the equality corpus of the espeak_ng processor (layers of the equality grade), seeded and frozen.

    python build_corpus.py --sources DIR --out DIR [--heldout]

DIR holds the public inputs: en_list (the vendor's word-list data file, pinned revision), the system word list,
the public-domain prose archive (gutenberg.zip), and for the frozen set the recorded 129 unknown-word texts
(unknown_words.json). Writes <layer>.txt (one line per row) and CORPUS.sha256 (sha256 of every layer file and
of the whole). Same inputs and seed give the same bytes. The held-out set uses another seed and other prose
books, and never repeats a line of the frozen set (pass --exclude DIR of it).
"""
import argparse
import hashlib
import json
import os
import random
import re
import zipfile

# Choices of the design (open questions; each a named place): layer sizes of section C, seeds, split of books.
LAYERS = [  # name, lines in the frozen set, lines in the held-out set (2 000 in all, same proportions)
    ("in_list", 6000, 600), ("rules", 6000, 600), ("affixes", 1500, 150), ("numbers", 1500, 150),
    ("abbreviations", 1000, 100), ("symbols", 800, 80), ("unknown", 700, 70), ("stress_context", 2500, 250),
]
SEED = {"frozen": 20261115, "heldout": 77310452}
BOOKS = {
    "frozen": ["austen-emma", "carroll-alice", "chesterton-thursday", "melville-moby_dick", "bible-kjv"],
    "heldout": ["austen-sense", "austen-persuasion", "chesterton-brown", "edgeworth-parents", "bryant-stories"],
}


# The corpus is public text: a line holding a word the public vocabulary rule bans is left out (written as codes, as the rule's own check does).
_W = ["".join(map(chr, c)) for c in ((102, 111, 114, 103, 101), (116, 114, 97, 99, 101))]
BANNED = re.compile(r"(?<![A-Za-z])(?:%s|(?:re)?%s(?:e|ed|es|er|ers|ing))(?![A-Za-z])" % (_W[0], _W[1][:-1]), re.I)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def list_keys(path):
    keys = []
    for ln in open(path, encoding="utf-8"):
        if ln.startswith(("//", "$", "#")) or not ln.strip():
            continue
        m = re.match(r"^([A-Za-z][A-Za-z'\-]*)\s", ln)
        if m:
            keys.append(m.group(1))
    return sorted(set(keys))


def sentences(zpath, books):
    z = zipfile.ZipFile(zpath)
    out = []
    for b in books:
        t = z.read("gutenberg/%s.txt" % b).decode("latin-1", "replace")
        t = re.sub(r"\[[^\]]*\]", " ", t)
        t = re.sub(r"\s+", " ", t)
        for s in re.split(r"(?<=[.!?])\s+", t):
            n = len(s.split())
            if 3 <= n <= 15 and s[:1].isalpha() and "_" not in s and "  " not in s:
                out.append(s.strip())
    return sorted(set(out))


def _pools(src, heldout):
    keys = list_keys(os.path.join(src, "en_list"))
    keyset = {k.lower() for k in keys}
    words = [w.strip() for w in open(os.path.join(src, "words"), encoding="utf-8")]
    return {
        "keys": keys,
        "plain": sorted({w for w in words if re.fullmatch(r"[a-z]{3,}", w) and w not in keyset}),
        "names": sorted({w for w in words if re.fullmatch(r"[A-Z][a-z]{2,}", w) and w.lower() not in keyset}),
        "prose": sentences(os.path.join(src, "gutenberg.zip"), BOOKS["heldout" if heldout else "frozen"]),
    }


def _invented(r):
    c, v = "bcdfghjklmnprstvwz", "aeiou"
    return "".join(r.choice(c) + r.choice(v) + (r.choice(c) if r.random() < .5 else "") for _ in range(r.randint(2, 4)))


def _rules(r, P, n):
    first, second = n * 6 // 10, n * 2 // 10
    return (r.sample(P["plain"], first) + r.sample(P["names"], second)
            + [_invented(r) for _ in range(n - first - second)])


def _affixes(r, P, n):
    suf = ["s", "ed", "ing", "ly", "er", "ness", "ment", "able"]
    a = [w + r.choice(suf) for w in r.sample(P["plain"], n // 2)]
    a += ["%s-%s" % (r.choice(P["plain"]), r.choice(P["plain"])) for _ in range(n // 4)]
    a += [r.choice(P["plain"]).capitalize() + r.choice(P["plain"]).capitalize() for _ in range(n - len(a))]
    return a


def _ordinal(r):
    k = r.randint(1, 999)
    return "%d%s" % (k, "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th"))


def _numbers(r, P, n):
    gens = [
        lambda: str(r.choice([r.randint(0, 99), r.randint(100, 9999), r.randint(10000, 10 ** 9)])),
        lambda: _ordinal(r),
        lambda: "%d.%d" % (r.randint(0, 999), r.randint(0, 9999)),
        lambda: str(r.randint(1100, 2099)),
        lambda: "%d:%02d" % (r.randint(0, 23), r.randint(0, 59)),
        lambda: "%d-%d" % (r.randint(1, 99), r.randint(100, 999)),
        lambda: "%d/%d" % (r.randint(1, 9), r.randint(2, 16)),
        lambda: "%s%s" % (r.choice("$£€"), "{:,.2f}".format(r.random() * r.choice([10, 1000, 10 ** 6]))),
        lambda: "%d %s" % (r.randint(1, 500), r.choice(["km", "kg", "ft", "mph", "%", "GB", "cm", "lbs"])),
        lambda: "%d%%" % r.randint(1, 100),
        lambda: "-%d" % r.randint(1, 9999),
    ]
    return [r.choice(gens)() for _ in range(n)]


def _one_abbreviation(r, P):
    titles = ["Mr.", "Mrs.", "Dr.", "Prof.", "St.", "Jr.", "Sr.", "etc.", "vs.", "e.g.", "i.e.", "Inc.", "Ltd.", "approx."]
    k = r.random()
    if k < .4:
        return "".join(r.choice("ABCDEFGHIJKLMNOPRSTUVWXYZ") for _ in range(r.randint(2, 5)))
    if k < .55:
        return r.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if k < .75:
        return "%s %s" % (r.choice(titles), r.choice(P["names"]))
    if k < .9:
        return "".join(r.choice([c, c.lower()]) for c in r.choice(P["plain"]).upper())
    return ".".join(r.choice("ABCDEFGHJKLMNPRSTUW") for _ in range(r.randint(2, 4))) + "."


def _one_symbol(r, P):
    sy = ["&", "@", "#", "+", "=", "*", "/", "\\", "~", "^", "<", ">", "|", "%", "$", "-", "--", "—", "…", "...", "“", "”", "‘", "’"]
    emo = ["😀", "🎉", "🚀", "❤️", "👍", "🌍", "😂", "🔥", "✅", "⭐"]
    paths = ["/usr/bin/", "C:\\Users\\", "~/", "./", "../", "http://", "www.", ".com", "_", "::", "->", "=>", "!=", "<=", "&&"]
    w = lambda: r.choice(P["plain"])  # noqa: E731
    k = r.random()
    if k < .25:
        return r.choice(paths) + w()
    if k < .45:
        return "%s %s %s" % (w(), r.choice(sy), w())
    if k < .6:
        return "%s %s" % (r.choice(emo), w())
    if k < .8:
        return '"%s," she said — "%s!"' % (w(), w())
    return "(%s) %s; %s: %s?" % (w(), w(), w(), w())


def _unknown(r, P, n, recorded):
    acc = ["café", "naïve", "résumé", "über", "señor", "façade", "Zoë", "jalapeño", "fiancée", "Möbius", "Ångström", "coöperate"]
    u = list(recorded)
    while len(u) < n:
        k = r.random()
        if k < .5:
            u.append("".join(r.choice("bcdfghjklmnpqrstvwxz") for _ in range(r.randint(3, 9))))
        elif k < .75:
            u.append(r.choice(acc) + ("" if r.random() < .5 else " " + r.choice(P["plain"])))
        else:
            u.append("".join(r.choice("bcdfgklmnprstaeiou") for _ in range(r.randint(5, 12))))
    return u


def build(src, heldout):
    r = random.Random(SEED["heldout" if heldout else "frozen"])
    P = _pools(src, heldout)
    size = {n: 3 * (h if heldout else f) for n, f, h in LAYERS}  # oversized: main() dedupes and cuts to the layer size
    recorded = [] if heldout else json.load(open(os.path.join(src, "unknown_words.json")))
    # the word list has fewer distinct keys than the design's 6 000: the held-out set takes a tenth, the frozen set the rest
    ks = random.Random(0).sample(P["keys"], len(P["keys"]))
    cut = len(ks) // 10
    return {
        "in_list": ks[:cut] if heldout else ks[cut:],
        "rules": _rules(r, P, size["rules"]),
        "affixes": _affixes(r, P, size["affixes"]),
        "numbers": _numbers(r, P, size["numbers"]),
        "abbreviations": [_one_abbreviation(r, P) for _ in range(size["abbreviations"])],
        "symbols": [_one_symbol(r, P) for _ in range(size["symbols"])],
        "unknown": _unknown(r, P, size["unknown"] + len(recorded), recorded),
        "stress_context": r.sample(P["prose"], min(size["stress_context"], len(P["prose"]))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--exclude", help="folder of a corpus whose lines the output must not repeat")
    a = ap.parse_args()
    L = build(a.sources, a.heldout)
    seen = set()
    if a.exclude:
        for n, _, _ in LAYERS:
            seen |= set(open(os.path.join(a.exclude, n + ".txt"), encoding="utf-8").read().split("\n"))
    os.makedirs(a.out, exist_ok=True)
    tot = hashlib.sha256()
    lines = []
    for n, _, _ in LAYERS:
        rows = []
        want = {x: (h if a.heldout else f) for x, f, h in LAYERS}[n]
        order = list(L[n])
        if n != "unknown":  # the recorded texts come first in "unknown" and must survive the cut
            random.Random(SEED["heldout" if a.heldout else "frozen"]).shuffle(order)
        for t in order:
            t = " ".join(t.split())
            if t and not BANNED.search(t) and t not in seen and len(rows) < want:
                rows.append(t)
                seen.add(t)
        data = ("\n".join(rows) + "\n").encode("utf-8")
        open(os.path.join(a.out, n + ".txt"), "wb").write(data)
        lines.append("%s  %s.txt  %d" % (sha(data), n, len(rows)))
        tot.update(data)
    lines.append("%s  ALL" % tot.hexdigest())
    open(os.path.join(a.out, "CORPUS.sha256"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
