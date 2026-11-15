# The processors place

A model's non-neural sub-components (a text front end, for one) are processors: declared by the model's container
(`manifest.json`, key `processors`, with the exact artifact and its sha256), delivered on demand from the hosts the
engine knows, verified, cached under the NeuroBrix cache, and called through one generic interface.

A processor holds NO tensor logic: it is standard-library text work, and never holds a tensor, a device or an array.
Anything tensor is nbxtensor's.

| file | role |
|---|---|
| `declaration.py` | the one reader of `manifest["processors"]`; every shape it refuses is named |
| `interface.py` | the call interfaces by `kind` (today `text_front_end`), what the engine lends (`ask`) |
| `store.py` | the door: find in the cache, fetch, verify sha256, lock, load; install a file |
| `messages.py` | every sentence a person reads about a processor, and nowhere else |
| `__init__.py` | `open_declared(manifest, container_dir, components)`: the entry processor, or None |

The commands: `neurobrix processor fetch <model> [--to <folder>]`, `neurobrix processor install <file> [--for
<model>]`, `neurobrix processor list`.

## The artifact

A processor's SOURCE is a sub-package here, `processors/<name>/` (its `processor.json`, its `LICENSE`, its code
folder, its tests, its `EQUALITY.md`). It never ships in the engine's wheel or sdist (one exclusion line per target in
`pyproject.toml`, held by `tests/unit/packaging/test_no_processor_sub_package_ships_in_the_wheel_or_the_sdist.py`):
the engine fetches its artifact on demand.

`python tools/build_processor.py <name> --out <folder>` builds ONE artifact, `<name>-<version>.nbxp`: a deterministic
zip (members sorted, stored, one fixed time and mode: the same sources give the same sha256 on any host) of two
separated trees, each with its licence file and its licence stated in `processor.json`:

| member | what |
|---|---|
| `processor.json` | the source's, plus `members`: the sha256 of every other member |
| `code/` | our code (the `code.source` folder, Python source of the standard library only) and `LICENSE` |
| `data/` | the pinned files (`data.files`: path, url, revision, sha256, size), unchanged; `data.licence_file` among them |
| `EQUALITY.md` | the equality report |

The build refuses, by file, line and name, an import beyond the standard library (or a dynamic import, or a file
that is not Python source) in the code tree, and a pinned file whose size or sha256 is not its pin.
