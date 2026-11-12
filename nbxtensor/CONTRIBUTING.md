# Contributing to nbxtensor

nbxtensor is open source under the Apache 2.0 license. Issues, fixes, new device files and certificates are welcome.

## Issues

Found a bug on your hardware? [Open an issue](https://github.com/NeuroBrix/nbxtensor/issues) with:

- your GPU model, its driver version, your operating system and Python version;
- `pip show nbxtensor triton` (the versions installed);
- the smallest code or command that shows the problem, and its full output.

## Pull requests

1. Fork the repository and branch from `main`.
2. Keep a change complete: the code, its test (seen failing before the fix, passing after), and one line under
   `## [Unreleased]` in `CHANGELOG.md` when a user would notice the change (a bold title, the cause and the effect).
3. Run the library's tests: `pip install -e ".[test]"` then `pytest tests`. On a machine with a GPU the examples of the
   README that launch a kernel run too.
4. Open the pull request.

The library's source lives in `nbxtensor/` of the [NeuroBrix repository](https://github.com/NeuroBrix/neurobrix) and is
exported here; a pull request on either repository is fine — we carry it across.

## The home of every tensor mechanism

**nbxtensor is the ecosystem's torch** (owner rule, 2026-11-12). Everything that touches tensors or follows torch's
logic (tensors, devices, memory, arenas, copies, compute kernels, replay, the tensor side of the experts store) lives in
nbxtensor, never in the engine. nbxtensor is a library with its own role: it replaces torch, and the engine is one of
its users, like any other. The engine keeps policy: Prism, model families, flows, the CLI, serving, which expert sits in
which slot and how many slots Prism grants. Every project of the ecosystem is separate (the engine, nbxtensor,
NBXGrammar, NBX Code, the container builder, the hub), and a function is filed in the project it belongs to. Engine code
that does tensor work is a defect as severe as a wrong answer: its fix ships at once, never held back to be grouped into
a release. The PyTorch engine computes with torch, its own library; that is not a breach.

For a contributor: a tensor helper is never added to the engine "for now"; the library is improved instead. A tensor
operation, a copy, a memory fact or a kernel the engine needs and nbxtensor lacks is added to nbxtensor, with its test,
and the engine calls it.

## Style

- **No torch.** nbxtensor never imports torch, not even optionally or inside a `try`.
- **No guessed value.** A fact the library needs (a tile, a tolerance, a limit of the device) is read from the device's
  file (`src/nbxtensor/backend/devices/<vendor>/<name>.yml`) or measured; a missing fact is an error that names it,
  never a default.
- **No model knowledge.** The library knows tensors, kernels and devices, never a model by name.
- English in code, comments and documentation.

## Contributor Assignment Agreement

Before your first pull request is merged, you sign the [NeuroBrix Contributor Assignment
Agreement](https://github.com/NeuroBrix/neurobrix/blob/main/docs/legal/contributor-assignment-agreement.md) — once, for
every NeuroBrix project (neurobrix, nbxtensor, NBXGrammar, NBX Code, the hub). You assign the rights in your contribution
to the owner of the NeuroBrix project (named in the NOTICE file) and its successors, and receive back a license to use
your own contribution as you wish; the owner may license the projects under other terms; no trademark right is involved.
To sign, post on your pull request the comment `I have read the NeuroBrix Contributor Assignment Agreement and I hereby
sign it.` — the `NeuroBrix CLA` check then passes. Contributing for a company? Your employer signs the [entity
agreement](https://github.com/NeuroBrix/neurobrix/blob/main/docs/legal/entity-contributor-assignment-agreement.md) first:
write to contact@neurobrix.org.

## Certificates

Certificates are the proven kernel settings of a GPU class: see [Certify your own hardware](README.md#certify-your-own-hardware)
for how to make them and share them. Open the pull request with the `certificate` template.
