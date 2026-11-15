"""Runs the vendor stack on the corpus and stores its outputs: the reference of the equality grade.

    <reference-env>/bin/python build_reference.py --corpus DIR --out DIR

The stack is phonemizer's espeak back end over libespeak-ng with the makers' options (stress kept, punctuation
kept, tie mark '^'), per variant en-us and en-gb, one line in, one unit sequence out. Writes
<variant>/<layer>.jsonl.gz (deterministic bytes) and REFERENCE.json (versions, library sha256, options,
corpus sha256 and the sha256 of every output file). Only the vendor's OUTPUTS are stored.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import sys

VARIANTS = {"us": "en-us", "gb": "en-gb"}
OPTIONS = {"preserve_punctuation": True, "with_stress": True, "tie": "^"}  # the makers' options
LAYERS = ["in_list", "rules", "affixes", "numbers", "abbreviations", "symbols", "unknown", "stress_context"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def gz(data):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, filename="") as f:
        f.write(data)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import espeakng_loader
    from importlib import metadata
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    lib = espeakng_loader.get_library_path()
    libsha = sha(open(os.path.realpath(lib), "rb").read())
    meta = {"stack": {"espeakng-loader": metadata.version("espeakng-loader"),
                      "phonemizer-fork": metadata.version("phonemizer-fork"),
                      "python": "%d.%d" % sys.version_info[:2],
                      "library": os.path.basename(os.path.realpath(lib)), "library_sha256": libsha},
            "options": OPTIONS, "variants": VARIANTS,
            "corpus_sha256": open(os.path.join(a.corpus, "CORPUS.sha256")).read().split()[-2:][0],
            "outputs": {}}
    for v, lang in VARIANTS.items():
        be = EspeakBackend(language=lang, **OPTIONS)
        os.makedirs(os.path.join(a.out, v), exist_ok=True)
        for layer in LAYERS:
            rows = open(os.path.join(a.corpus, layer + ".txt"), encoding="utf-8").read().split("\n")[:-1]
            out = []
            for t in rows:
                ps = be.phonemize([t])
                out.append(json.dumps([t, ps[0] if ps else ""], ensure_ascii=False))
            data = gz(("\n".join(out) + "\n").encode("utf-8"))
            p = os.path.join(a.out, v, layer + ".jsonl.gz")
            open(p, "wb").write(data)
            meta["outputs"]["%s/%s.jsonl.gz" % (v, layer)] = sha(data)
    blob = json.dumps(meta, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    open(os.path.join(a.out, "REFERENCE.json"), "w", encoding="utf-8").write(blob)
    print(sha(blob.encode("utf-8")), "REFERENCE.json")


if __name__ == "__main__":
    main()
