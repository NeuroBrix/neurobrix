# Census once, certify once, run only to confirm

The owner's method (2026-09-28). The slowness of a run is the autotune, and the autotune is done
**without running models**.

## The three records

| record | where | what it holds | who writes it |
|---|---|---|---|
| the census table | `src/neurobrix/config/census/<vendor>/<profile>/<class>g.jsonl` | every kernel key each container's runs form: model, container sha, mode, rungs, op, kernel, key, dtype, tool revision | the census tool, from the containers alone (a shadow run: no weights, no device) |
| the certified directory | `src/neurobrix/config/autotune/<vendor>/<profile>/` | one proven configuration per key and memory class, with its proof | the certifier, from the census table and nothing else |
| the matrix | the confirmation runs' rows | one verdict per model × mode × dtype | a run in certified-only mode |

A container that changes (a retrace, a naming pass) has its census rows **replaced**, never added
to; a kernel whose key definition changes has every row regenerated. A census is fast or it is
broken: it enumerates shape classes, never a model's generation step by step.

## Certified-only

Every matrix, gate and verification run passes `neurobrix run --certified-only`
(`NBX_AUTOTUNE_CERTIFIED_ONLY=1`). A key the certified directory does not serve for the card's
memory class **fails the run**, naming the key and what the census table says about it:

* **absent from the table** — a census defect: the census tool is fixed, the table regenerated,
  the key certified, the cell run again;
* **present in the table, not certified** — a certification gap: the table is certified, the cell
  run again.

The machine's local replay cache is neither read nor written in this mode: it holds earlier runtime
sweeps, not certifications. A cell that is slow in this mode is a real engine cost.

## A model runs whole only to confirm

After its trace is correct, its census taken and its keys certified — with the smallest request that
still judges it. Diagnosis uses op-level probes and the vendor's own pipeline as the oracle, never
a full-model sweep.
