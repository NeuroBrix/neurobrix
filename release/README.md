# Release manifests

One file per release tag, `<tag>.sha256`: the sha256 and name of every distribution file the release gate trialled
(`sha256sum` format). It is committed BEFORE the tag. The publish workflow (`.github/workflows/publish.yml`) uploads the
release's files only when `tools/release_files_check.py` finds them byte-identical to this list.
