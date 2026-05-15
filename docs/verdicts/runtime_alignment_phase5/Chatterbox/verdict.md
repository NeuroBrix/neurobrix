# Chatterbox — verdict

**Verdict agent**: PASS_STRUCTURE_ONLY
**Family**: tts  •  **Mode**: audio
**Duration**: 21.4s  •  **Exit**: 0

**Relaunch**:
```
/home/mlops/ml/venv/bin/neurobrix run --model chatterbox --output /home/mlops/NeuroBrix_System/validation_outputs/runtime_alignment_phase5/Chatterbox/output.wav --prompt Hello world
```

Hocine validation: TODO


## Structure-only note (Hocine manual review)
Wav structure is healthy (10.86s, RMS 0.144, non-silent) so the family dispatch correctly produced an audio file via save_output. Audio QUALITY is charabia per Hocine — this is an internal Chatterbox decoding-head bug, out of scope of the 9-family alignment chantier.
