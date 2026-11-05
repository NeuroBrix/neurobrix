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

## Style

- **No torch.** nbxtensor never imports torch, not even optionally or inside a `try`.
- **No guessed value.** A fact the library needs (a tile, a tolerance, a limit of the device) is read from the device's
  file (`src/nbxtensor/backend/devices/<vendor>/<name>.yml`) or measured; a missing fact is an error that names it,
  never a default.
- **No model knowledge.** The library knows tensors, kernels and devices, never a model by name.
- English in code, comments and documentation.

## Certificates

Certificates are the proven kernel settings of a GPU class: see [Certify your own hardware](README.md#certify-your-own-hardware)
for how to make them and share them. Open the pull request with the `certificate` template.
