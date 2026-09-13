# Census retirements

Keys removed from a machine's replay-cache census because the certifier declared them UNREACHABLE — recorded by an older engine whose wrappers computed the key differently. One dated paragraph per retirement; append, never edit. Tool: `tools/census_retire_unreachable.py`.

## 2026-09-13 01:07 UTC — 183 keys retired, engine `17a05e0`

Named UNREACHABLE by `neurobrix autotune certify` (the wrapper computed a different key for inputs synthesised from the census key) in: `certify_baddbmm.log`, `certify_matmul.log`, `certify_addmm_conv.log`.

| kernel | retired |
|---|---:|
| `addmm_kernel` | 7 |
| `baddbmm_kernel` | 113 |
| `matmul_kernel` | 63 |

Census 7314 → 7131 keys. Reversible record: `/home/mlops/nbx/campaigns/2026_09_12_night_catalogue/census_retired_20260913_010727.json`; backup: `autotune_configs_cuda-70.json.bak.20260913_010727`.

## 2026-09-13 01:23 UTC — 2 keys retired, engine `06004e2`

Named UNREACHABLE by `neurobrix autotune certify` (the wrapper computed a different key for inputs synthesised from the census key) or FAILED (inputs that cannot be synthesised) in: `certify_resume_card0.log`.

| kernel | retired |
|---|---:|
| `conv2d_forward_kernel` | 2 |

Census 7131 → 7129 keys. Reversible record: `/home/mlops/nbx/campaigns/2026_09_12_night_catalogue/census_retired_20260913_012345.json`; backup: `autotune_configs_cuda-70.json.bak.20260913_012345`.
