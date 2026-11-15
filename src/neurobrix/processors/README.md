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
