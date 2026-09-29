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

## The certifier

`src/neurobrix/kernels/autotune_certify.py`, `autotune_certified.py`, `cli/commands/autotune.py`
(`neurobrix autotune certify`). Contract commit 100ed7af (release-candidate-1). Test:
`tests/unit/kernels/test_the_certifier_holds_its_contract.py`.

| # | clause | how it holds |
|---|---|---|
| 1 | one key function | the launcher and the certifier form the key with `key_of`; the certifier refuses a table key the wrapper cannot form (`UnreachableCensusKey`). **OPEN:** no CPU round trip from the loop through filing to `lookup`/`apply` |
| 2 | two writers | `_write_file` under `<file>.json.lock`, the disk's file as base; `restamp` under the same lock — test: two processes stalled between read and write; a restamp blocked by a held lock |
| 2 | a corrupt file | refused (`UnreadableCertifiedFile`), never read as empty and overwritten |
| 3 | idempotence | entries written sorted — test: the same proof twice, same bytes. **OPEN:** an end-to-end `--only-missing` pass on a complete directory (needs a card) |
| 4 | inputs | `--kernels` naming nothing or an unknown kernel, an empty table, an unparsable key: refused by name |
| 5 | exits | `sigterm_ends_through_finally`: SIGTERM ends a pass through the writer's flush — test: a child process sent SIGTERM |

## The checkpointer

`tools/certified_checkpoint.py`. Contract commit 4c3a857f (release-candidate-1). Test:
`tests/unit/tools/test_the_checkpointer_holds_its_contract.py` with the four existing suites.

| clause | how it holds |
|---|---|
| the committed bytes are the gated bytes | the changed files read once, a copy gated, those blobs committed through a temporary index (the repository's hooks run); the real index reset for those paths |
| main is never pushed | refused inside `checkpoint()`, on the branch at push time |
| one per repository | an exclusive lock in the common git dir for the process's life; a second instance refused by name |
| at most one push per 30 minutes | every ATTEMPT spends the window; the stamp replaced atomically |
| an input that names nothing | a `--dir` that does not exist refused |

## The gate harness

`tools/regression_matrix.py`. Contract commit d10cec52 (release-candidate-1 828e9fd6). Test:
`tests/unit/tools/test_the_gate_harness_holds_its_contract.py` (18 tests, each red on its
injection) with the 11 existing harness suites.

| clause | how it holds |
|---|---|
| a miss is counted and named | each row: `misses` (distinct KeyNotCertified), `first_missing_key`; `first_error` names the miss over its inner error |
| inputs | a blank model or mode, an unknown mode, a model absent from the cache, an empty cache refused before any cell; a cell asked for with no row is an error naming it |
| the committed reference | each row records `tree_dirty` over the `--src` tree's config/autotune and config/census, and `certified_dir_override`; the override refuses the run unless `--allow-certified-dir-override`; an earlier row is reused only at the same engine sha with a clean reference on both sides |
| readable files | a cut last line skipped and said, a malformed inner line refused; appends fsynced, a cut fragment moved aside; the table and the export written atomically. **OPEN:** `host_ledger.json` / `host_waiting.json` rewritten in place under their flock |
