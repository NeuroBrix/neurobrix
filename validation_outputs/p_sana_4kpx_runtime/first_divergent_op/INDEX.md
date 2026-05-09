# P-SANA-4KPX-RUNTIME / first_divergent_op

R29 inspectable artefact for Hocine pivot 2026-05-09:
"identify the FIRST op where sequential and triton_sequential
diverge significantly on full-tensor metrics".

| file | content |
|---|---|
| [walk_table.md](walk_table.md) | full chronological walk + bisection + verdict |
| walk_stats.json | per-op JSON stats from `/tmp/walk_ops_full_diff.py` |

| field | value |
|---|---|
| chantier | P-SANA-4KPX-RUNTIME |
| date | 2026-05-09 |
| first divergent op | `aten.add::0` (op_idx=5) |
| divergence type | dtype propagation mismatch (seq fp16 vs tri fp32) |
| algorithmic kernel bug? | NO (NBX add bit-exact vs torch add on fp16 random) |
| root cause area | metadata chain dtype handling: DtypeEngine vs TritonDtypeEngine |
| Hocine arbitrage | TODO — three fix options proposed in walk_table.md |

## R29 status — Hocine inspection

- agent verdict: localized at add::0 dtype propagation
- recommend Hocine: read walk_table.md "Awaiting Hocine arbitrage"
  section, decide fix path before any code change

## Reproduce

```bash
NBX_DISABLE_AUTOTUNE=1 python3 /tmp/walk_ops_full_diff.py
NBX_DISABLE_AUTOTUNE=1 python3 /tmp/capture_add0_inputs.py
NBX_DISABLE_AUTOTUNE=1 python3 /tmp/microtest_add0.py
```

(Saved latent prerequisite: `vae_isolation_input.pt` from prior
sequential 4Kpx capture.)
