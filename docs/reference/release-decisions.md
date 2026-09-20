# Release decisions — items that shape a release, recorded where a release is decided

One entry per item, dated, with the measurement that raised it. A campaign directory holds
the evidence; this file holds the decision the release has to make. Closed items move to the
CHANGELOG entry that closed them.

## 0.5.5

### A compiler upgrade inherits a directory proven under the previous compiler (2026-09-20)

**Measured.** The certified autotune directory `nvidia/volta` shipped in 0.5.4 records
Triton 3.6.0 in every one of its 9 655 proofs. The generator gate (`ec938641`) serves a
certified setting only to the compiler that proved it, which is right. So an installation
that upgrades to Triton 3.8.0 — which `pip install torch==2.14` does by itself — gets a
first run in which **every kernel shape sweeps at execution time**: on this rack, a
regression battery that took 64 minutes served ran on a four-hour trajectory unserved
(the replay cache grew 256 → 1 115 entries in 2 h 30; upscaler cells 12 s → 102–113 s).
Nothing warns; the run is merely slow, and the settings it sweeps land in the local replay
cache, which records no compiler version at all and therefore serves them to the next
compiler as well.

**The decision 0.5.5 has to make**, one of:

1. **Ship the directory for both compilers** — the `variants` slot already composes one
   entry per memory class; a generator dimension beside it lets one file carry the 3.6.0 and
   the 3.8.0 proofs, and the gate picks by the running compiler. Re-proof is a night of four
   cards per compiler (measured 2026-09-16/17: 10 087 entries).
2. **Name the upgrade path** — `neurobrix autotune certify --reprove-generator` on the
   machine, said by `neurobrix doctor` and by the engine at the first refused entry
   ("certified under triton 3.6.0, running 3.8.0: N entries will sweep; re-prove with …"),
   so the sweep is a choice the user made rather than a slowdown nobody explained.

The second is cheap and honest and should ship regardless; the first is what makes an
upgrade free. Either way the **replay cache must record the compiler** it swept under and
serve only to that one — the directory gate without the cache gate is half a door.

### The engine under a `CUDA_VISIBLE_DEVICES` mask (2026-09-20, measurement pending)

A user in a container, on a shared server or under a scheduler runs masked as the normal
case. Three regression guards (`test_serve_warm.py`, `warm_cell_runner.py`,
`test_upscale_offtrace.py`, 2026-08-27) and debt entry D-AUTODETECT-VISIBLE-MASK say the
engine mis-places under a mask; `_apply_visible_filter` (`acd14637`, 2026-09-03) re-indexes
the visible set and the profile is keyed by it since. Whether a facet remains — the profile
index against the runtime ordinal — is being measured with one pinned cell, red on the tree
before the fix and then green on main. If a facet remains it is an engine defect that 0.5.5
must close, not a harness inconvenience; if none does, the guards go to the vacuous-gates
register and the skips come off.
