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

1. A lot that changes what a user of a product sees (a behaviour, a speed, a result, a refusal, a public name) adds
   its line under `[Unreleased]` of THAT product's changelog, in the same commit: a bold title, the cause and the
   effect, in Added / Changed / Deprecated / Removed / Fixed / Security. — whoever lands it.
2. The landing script refuses a lot that touches a product's code without touching its changelog, unless the lot is
   declared to change no behaviour, with its reason, which the landing journal records. — the landing script.

## Publishing a version — after the owner's GO

1. The owner says GO, naming the product and the version. — the owner.
2. The tree to publish is the one the final gate ran on, frozen; its proofs are cited in the release decision
   (`docs/reference/release-decisions.md`). — the Dell.
3. In the product's changelog, `[Unreleased]` becomes `[X.Y.Z] - YYYY-MM-DD` and a new empty `[Unreleased]` opens above
   it; the comparison links at the bottom gain the new version. The product's version field is set to X.Y.Z; for the
   library, the engine's dependency (`nbxtensor==X.Y.Z` in `pyproject.toml`) follows it in the same commit. One commit,
   landed like any other. — the Dell.
4. The tag (`vX.Y.Z` or `nbxtensor-vX.Y.Z`) is made on that commit and pushed to both remotes, read back. — the Dell.
5. The GitHub Release of the tag is created with the version's changelog section as its body — or, where the section
   exceeds GitHub's 125 000 characters (the engine's 0.6.0: 246 267), its summary, the install lines and a link to the
   section; the GitLab release mirrors it. — the Dell.
6. The package is built from the tag and uploaded to PyPI; its `[project.urls] Changelog` points at the changelog, and
   its README links it. — the Dell.
7. The two installables are installed on a clean machine from PyPI and run through the entry point (the stable's
   check). — the Dell and the Mac.
8. The hub synchronises the release (notes, archive, download) and publishes the notes the same day, on the product's
   release page. — the hub.
9. The release is recorded in `docs/reference/release-decisions.md` with its proofs and the GO's time. — the Dell.
