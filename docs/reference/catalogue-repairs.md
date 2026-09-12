# Catalogue repairs — containers fixed at the source and delivered

A repair is not a patch and not a diagnosis. It is one motion, and every step of
it is recorded here because a re-trace without a re-upload is a forbidden state
(CLAUDE.md, hub freshness) and a container that only exists locally has not been
delivered.

**The five steps, and a repair is closed only when all five carry a date:**

1. **Cause** named at the source — Forge, never a runtime compensation.
2. **Re-trace** producing a new graph, with its op count.
3. **Rebuild**, with the container size beside the previous one.
4. **Re-upload**, reusing the exact slug so the record, its counters and its
   creation date survive.
5. **Proof by run** — the model producing output from the installed container.
   Not a reasoning about why it should. A verification that has not looked at the
   real artefact says nothing, and this project paid for that twice on the day
   this file was created.

A repair missing step 5 is listed as **delivered, unproven**, never as closed.

---

## 1 — `rhymes-ai/Allegro-TI2V` · 2026-09-12

**The first container of the catalogue repaired end to end and delivered.**

| step | evidence |
|---|---|
| cause | `RuntimeExecutor._container_output_size` refused every container whose flow was not `iterative_process`, and never consulted the VAE when the backbone's latent was flattened. Allegro-TI2V died on `Key 'latent_height' not found in runtime/defaults.json`. Fixed at the resolution engine, not in the model: a decoder's input extents ARE latent extents and the container states them. |
| re-trace | not required — the defect was in the runtime's reading of the container, not in the graph |
| rebuild | 2026-09-12 14:06, 26.1 GB written |
| re-upload | 2026-09-12 15:13:48, `forge replace` onto the existing slug; hub shows 24.3 GB, record preserved |
| install | 2026-09-12 15:14:41, `forge local --overwrite` |
| verification | `_container_output_size` reads **(144, 208)** from the shipped container — measured inside the installed form, not inferred from the diff |
| proof by run | **owed**. Two attempts failed for reasons that were about the harness, not the container: pinned to a 16 GB card (OOM at `convolution::0` asking 9 GB for a 19 GB model), then to `"2,3"` which remapped the ordinals into a sticky `cudaErrorIllegalAddress`. Register entry 39. The unpinned run is the one that counts. |

The local archive was deleted after the hub copy was verified; the installed form
is a separate tree and survives. Both live on the NAS, not on the root
filesystem — a distinction that cost a wrong wall in a report the same day.

---

## What this file is for

The catalogue is 56 local containers and the hub is the distribution
destination. When a defect is found in one of them, the interesting question six
months later is not "was it diagnosed" — the register and the verdicts hold
that. It is **which containers people are actually being served**, and whether
the fix reached them. That is one table, and it has to be written at the moment
the upload lands or it is written from memory.
