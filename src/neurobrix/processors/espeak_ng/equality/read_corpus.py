"""Reads the corpus with OUR reader and writes the candidate folder the grade compares with the reference.

    python read_corpus.py --data DIR --corpus DIR --out DIR [--layers in_list ...]

--data is the folder of the pinned vendor data files under their original relative paths (never committed).
A line the reader does not read is written with a null reading (the grade counts it unequal) and its reason goes
to <out>/<variant>/<layer>.unread.txt.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reader import Reader, Unreadable, english  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", nargs="+", default=["in_list"])
    a = ap.parse_args()
    for variant in sorted(english.VARIANT_LANGUAGE):
        reader = Reader(a.data, variant)
        os.makedirs(os.path.join(a.out, variant), exist_ok=True)
        for layer in a.layers:
            with open(os.path.join(a.corpus, layer + ".txt"), encoding="utf-8") as f:
                texts = f.read().split("\n")[:-1]
            rows, unread = [], []
            for t in texts:
                try:
                    units = reader.units(t)
                except Unreadable as e:
                    units = None
                    unread.append("%s\t%s" % (t, e))
                rows.append(json.dumps([t, units], ensure_ascii=False))
            with open(os.path.join(a.out, variant, layer + ".jsonl"), "w", encoding="utf-8") as f:
                f.write("\n".join(rows) + "\n")
            with open(os.path.join(a.out, variant, layer + ".unread.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(unread) + ("\n" if unread else ""))
            print(variant, layer, len(texts), "lines,", len(unread), "not read")


if __name__ == "__main__":
    main()
