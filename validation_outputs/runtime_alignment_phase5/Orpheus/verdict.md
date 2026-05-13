# Orpheus — verdict

**Verdict agent**: FAIL_HORS_SCOPE_HORS_SCOPE_HORS_SCOPE
**Family**: tts  •  **Mode**: audio
**Duration**: 10.6s  •  **Exit**: 1
**Reason**: RuntimeError: Failed at op aten._scaled_dot_product_efficient_attention::0 (aten::_scaled_dot_product_efficient_attention): shape '[1, 8, 3, 0, 1]' is invalid for input of size 24

**Relaunch**:
```
/home/mlops/ml/venv/bin/neurobrix run --model orpheus-3b-0.1-ft --output /home/mlops/NeuroBrix_System/validation_outputs/runtime_alignment_phase5/Orpheus/output.wav --prompt Hello world
```

Hocine validation: TODO


Hors-scope: independent GQA wrapper bug, dedicated follow-up.
