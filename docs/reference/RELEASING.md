# Releasing

How a version of either product is published. This file IS the release workflow: one step per line, who does it, in
order. **The owner's GO is the gate of every tag and every publication** — no step below the gate is taken without it.

Two products, one repository:

| product | changelog | tag | package | public repository |
|---|---|---|---|---|
| the engine (`neurobrix`) | `CHANGELOG.md` | `vX.Y.Z` | PyPI `neurobrix` | GitHub `NeuroBrix/neurobrix`, GitLab mirror |
| the library (`nbxtensor`) | `nbxtensor/CHANGELOG.md` | `nbxtensor-vX.Y.Z` | PyPI `nbxtensor` | GitHub `NeuroBrix/nbxtensor` (the export of `nbxtensor/`) |

Both changelogs follow [Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning 2](https://semver.org/spec/v2.0.0.html).

## While developing — every landing

0. **Published numbers are consecutive, and main carries exactly one number per product: the next one to publish**
   (the last published + 1 — the owner, 2026-11-11). A publication increments the version; a fix does not: fixes and
   patches are grouped on main under that one number, in the product's single open changelog entry (`[Unreleased]`),
   organised by what a user gains. A lot touches neither a version field nor a pin, and no text or commit message names
   a number that was never published. A version goes out when the batch is consistent, the engine stable and the known
   defects fixed; the supervisor arbitrates when.

1. A lot that changes what a user of a product sees (a behaviour, a speed, a result, a refusal, a public name) adds
   its line under `[Unreleased]` of THAT product's changelog, in the same commit: a bold title, the cause and the
   effect, in Added / Changed / Deprecated / Removed / Fixed / Security. — whoever lands it.
2. The landing script refuses a lot that touches a product's code without touching its changelog, unless the lot is
   declared to change no behaviour, with its reason, which the landing journal records. — the landing script.
3. **A public page never says how a container is made.** Every tracked Markdown page speaks of "building containers"
   and "the container builder": it never names the private build tool and never uses its verb. A lot that touches a
   page runs `python tools/public_vocabulary.py` (red on those words, file and line); the only occurrences it accepts
   are the real names and quoted messages listed with their reason in `tools/public_vocabulary.allow.tsv`, a list that
   only shrinks. The unit test `tests/unit/tools/test_public_pages_never_say_how_a_container_is_made.py` holds it.
   — whoever lands it.

## Publishing a version — after the owner's GO

1. The owner says GO, naming the product and the version. — the owner.
2. The tree to publish is the one the final gate ran on, frozen; its proofs are cited in the release decision
   (`docs/reference/release-decisions.md`). — the Dell.
3. In the product's changelog, `[Unreleased]` becomes `[X.Y.Z] - YYYY-MM-DD` and a new empty `[Unreleased]` opens above
   it; the comparison links at the bottom gain the new version. X.Y.Z is the number main already carries (rule 0): the
   version field and the engine's dependency on the library (`nbxtensor==X.Y.Z` in `pyproject.toml`) are checked, not
   changed. One commit, landed like any other. Right after the publication, one commit sets each published product's
   field (and the pin) to its next number. — the Dell.
4. **Before the tag, the new user's path, on the BUILT wheels (never on PyPI), all green — or nothing is tagged:**
   (L) a clean Linux container on a free card: `pip install` of the built engine and library wheels, `neurobrix doctor`,
   `neurobrix import TinyLlama/TinyLlama-1.1B-Chat-v1.0 --accept-license` onto the container's own disk (no mount),
   `neurobrix run` with NO mode flag and its output read; the same path for a model of another family (Sana 1024,
   whisper-large-v3-turbo); then the same with `"neurobrix[pytorch]"`. (M) the Mac, the same path in a clean venv, with
   `python -m nbxtensor install-metal`, the Triton engine on Metal, the output looked at. Zero Traceback, zero wrong
   statement from `doctor` (the owner via the supervisor, 2026-11-07 02:48: 0.6.0's first user run ended in a
   Traceback that this path, run after the publication, found too late). — the Dell and the Mac.
5. The tag (`vX.Y.Z` or `nbxtensor-vX.Y.Z`) is made on that commit and pushed to both remotes, read back. — the Dell.
6. The GitHub Release of the tag is created with the version's changelog section as its body — or, where the section
   exceeds GitHub's 125 000 characters (the engine's 0.6.0: 246 267), its summary, the install lines and a link to the
   section; the GitLab release mirrors it. — the Dell.
7. The package is built from the tag and uploaded to PyPI; its `[project.urls] Changelog` points at the changelog, and
   its README links it. — the Dell.
8. The two installables are installed on a clean machine from PyPI and run through the entry point (the stable's
   check). — the Dell and the Mac.
9. The hub synchronises the release (notes, archive, download) and publishes the notes the same day, on the product's
   release page. — the hub.
10. The release is recorded in `docs/reference/release-decisions.md` with its proofs and the GO's time. — the Dell.
