# Equality grade tool of the espeak_ng processor

The equality grade of the processor: exact equality of the unit sequence, line by line, per variant (en-us, en-gb),
between a reader and the vendor stack run as a binary (phonemizer's espeak back end over libespeak-ng 1.52.0,
stress kept, punctuation kept, tie mark `^`).

What is in the repository: only our code. `grade.py`, `build_corpus.py`, `build_reference.py`, and `archive_pin.py`
(name, size, sha256 of the data archive). No layer of the corpus is ours: every one is drawn from the vendor's
word-list data, so the frozen corpus (8 layers, about 20 000 lines, `CORPUS.sha256`) and the vendor's stored outputs
for it (`reference/<variant>/<layer>.jsonl.gz`, `REFERENCE.json`) are in ONE archive outside git and outside the
package, `espeak_ng_equality_<version>.tar.zst`, with its own LICENSE (GPL-3.0-or-later) and a `MANIFEST.sha256`.

- Grade: `python grade.py --candidate DIR --archive FILE [--report EQUALITY.md]` (or `--data-dir DIR` holding the
  pinned archive). The archive is refused by name when absent or when its size or sha256 differs from the pin; the
  unpacked files are checked against the MANIFEST. Every unequal line is listed with both readings; exit 0 only at
  all lines equal. `--reference DIR` reads an unpacked folder (held-out set, test corpus).
- Read the corpus with our reader (the candidate folder of the grade): `read_corpus.py --data DIR --corpus DIR
  --out DIR`, where `--data` holds the pinned vendor data files under their original relative paths.
- Rebuild the corpus: `build_corpus.py --sources DIR --out DIR` (seeded; inputs: the vendor's word-list data file at
  its pinned revision, a system word list, a public-domain prose archive, the recorded unknown-word texts).
- Rebuild the reference: in a venv with `espeakng-loader==0.2.4` and `phonemizer-fork==3.3.2` installed (print the
  library's sha256 with `sha256sum $(python -c 'import espeakng_loader as e;print(e.get_library_path())')`), run
  `python build_reference.py --corpus corpus --out DIR2`.
- A held-out set of the same layers (other seed, other books) is kept outside the repository and read only by the
  grader.
- Tests: `tests/unit/processors/` grade a tiny corpus written for the tests, and check the packaged files.

Clean room: only the vendor binary's outputs are stored in the archive; no vendor source was read.
