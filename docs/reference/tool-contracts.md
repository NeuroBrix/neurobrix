# Tool contracts — the census, the certifier, the gate

The owner, 2026-09-29: many of this project's problems came from the tools that census, certify
and gate, not from the engine. Each tool is an engine too: it has a written contract, and every
clause has a test that was seen RED on a deliberate injection. One implementation serves both
machines. A clause without a test is written here as **OPEN**, with the reason.

The clauses shared by every tool:

1. **One key function.** The key a tool records, files or looks up is the key the launcher forms
   at run time — the same function, never a copy.
2. **Two writers lose nothing.** Every read-modify-write of a shared file holds that file's lock.
3. **Idempotence.** The same input twice leaves the same bytes.
4. **An input that names nothing is refused by name.** Empty, mis-globbed or absent inputs are
   refused before any work, never read as "everything" or "nothing to do".
5. **Every exit leaves a readable file.** Writes go beside the file and are replaced atomically;
   an exit through an exception leaves a report that says it was cut.
6. **A gate compares against the committed reference**, never against its own earlier output or
   a working copy.

## The census

`tools/certified_census.py` (the walk), `tools/derived_census.py table` (the derivation),
`src/neurobrix/kernels/census_table.py` (THE table: `read`, `write`, `locked`, `replace_model`),
`src/neurobrix/kernels/census.py` (the in-process recorder). Tests:
`tests/unit/tools/test_the_census_tool_holds_its_contract.py` (C below),
`tests/unit/kernels/test_the_census_is_one_table.py`,
`tests/unit/kernels/test_the_census_shadow_records_the_launchers_key.py`,
`tests/unit/census/test_the_derived_table_is_the_walked_census.py`,
`tests/unit/census/test_the_derivation_takes_the_flash_head_dim_detour.py`.

| # | clause | how it holds | test |
|---|---|---|---|
| 1 | the recorded key is the launcher's | the walk records the very key object the launcher looks up (`ops/_configs.py`: `key_of` -> `census.record` -> `certified.apply`/`refuse_missing`); the derivation forms its keys through `launch_keys`, whose routing functions the wrappers call (`sdpa_route`, `flash_headdim_detour`, `mm_dtypes`, ...) | the shadow test; derived = walked (TinyLlama); the detour test. **OPEN:** the derivation's key TUPLES are built in `launch_keys`, not by `key_of` over the kernel's `keys=` — a derived-vs-walked comparison exists for the LM matmul/attention family only (conv, depthwise, LSTM, DFT, tiled conv have none) |
| 2 | two writers lose nothing | `census_table.locked` (flock on `<table>.lock`) around `replace_model`, `consolidate`, `migrate` | C: two processes stalled between read and write |
| 3 | idempotence | `write` sorts, deduplicates and canonicalises | C: byte identity across two `replace_model` calls |
| 4 | a model's rows are never replaced by none | `replace_model` refuses an empty row list (`EmptyCensus`); the walk marks a model whose shadows formed no key `no_keys` and keeps its rows; the derivation refuses a model it derived no key for | C (both tools) |
| 4 | inputs refused by name | an empty `--models` entry, a model absent from the cache, an empty cache, an empty `--modes`; the derivation's empty `--modes`/`--rungs`/`--models` | C |
| 5 | readable on every exit | `write`: temporary in the table's directory, fsync, `os.replace`, the temporary removed on failure; the walk's `--out` written atomically, and on an exception a report that says `interrupted`; a key record's last line without its newline (a shadow cut mid-write) is dropped, never read as a key | C |

Row columns: model, container (a hash of the components' graph.json — **OPEN:** topology.json,
where the tiling probe lives, is not hashed), mode, rungs_mb, ops, kernel, key, dtype, tool
(the walk marks a dirty tree `+`; **OPEN:** the derivation does not).

## The certifier — next

The certifier's audit (2026-09-29) is written up; its fixes follow in this order: the lock
contended by two processes; a corrupt certified file refused, never overwritten; `--kernels`
empty or matching nothing refused; an empty census table refused; SIGTERM flushes the pending
proofs; `--only-missing` on a complete directory leaves identical bytes; the migration's two
empty-glob refusals tested.

## The gate — after the certifier

The gate harness (`tools/regression_matrix.py`) records per cell a miss count and the first
missing key; refuses an empty model list and a list that does not match the lists it was given;
never takes an earlier row from another engine sha as done; records whether the certified
directory it read was the committed one. The checkpointer refuses main inside `checkpoint()`,
refuses a second instance, and counts a failed push against its window; its dead
30-minute test file is removed or ported.
