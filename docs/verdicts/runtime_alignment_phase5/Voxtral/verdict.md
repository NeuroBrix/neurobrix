# Voxtral — verdict

**Verdict agent**: FAIL_HORS_SCOPE
**Family**: audio_llm  •  **Mode**: text
**Duration**: 29.7s  •  **Exit**: 0

**Relaunch**:
```
/home/mlops/ml/venv/bin/neurobrix run --model Voxtral-Mini-3B-2507 --output /home/mlops/NeuroBrix_System/validation_outputs/runtime_alignment_phase5/Voxtral/output.txt --audio /home/mlops/NeuroBrix_System/test_speech_ref.wav --prompt what is being said? --max-tokens 40
```

Hocine validation: TODO


## Hors-scope note
Voxtral hallucinates a reply unrelated to the input audio (Whisper transcribes the same wav perfectly, Voxtral returns 'I'm sorry I didn't quite catch that...'). Root cause is the audio_llm processor multimodal pipeline that doesn't feed the audio into the chat-template the way Voxtral expects. Tracked as a dedicated processor-multimodal follow-up; not caused by the 9-family dispatch refactor.
