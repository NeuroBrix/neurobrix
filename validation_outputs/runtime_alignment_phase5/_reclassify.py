"""Apply Hocine manual inspection feedback to verdicts."""
import json
from pathlib import Path

ROOT = Path("/home/mlops/NeuroBrix_System/validation_outputs/runtime_alignment_phase5")

PATCHES = {
    "Voxtral": {
        "verdict": "FAIL_HORS_SCOPE",
        "hors_scope_reason": (
            "Voxtral hallucinates a reply unrelated to the input audio "
            "(Whisper transcribes the same wav perfectly, Voxtral returns "
            "'I'm sorry I didn't quite catch that...'). Root cause is the "
            "audio_llm processor multimodal pipeline that doesn't feed the "
            "audio into the chat-template the way Voxtral expects. Tracked "
            "as a dedicated processor-multimodal follow-up; not caused by "
            "the 9-family dispatch refactor."
        ),
    },
    "Chatterbox": {
        "verdict": "PASS_STRUCTURE_ONLY",
        "structure_only_reason": (
            "Wav structure is healthy (10.86s, RMS 0.144, non-silent) so "
            "the family dispatch correctly produced an audio file via "
            "save_output. Audio QUALITY is charabia per Hocine — this is "
            "an internal Chatterbox decoding-head bug, out of scope of the "
            "9-family alignment chantier."
        ),
    },
    "Janus-image": {
        "verdict": "PASS_STRUCTURE_PARTIAL",
        "structure_only_reason": (
            "Image is structurally coherent (384x384, mean=121.9, std=62.8) "
            "and shows a real cat (correct semantic). Color fidelity off — "
            "prompt 'a red cat' produced an orange/ginger cat. Likely model "
            "sampling / CFG issue, not a runtime dispatch bug. Color fidelity "
            "follow-up may need higher CFG or different prompt; out of scope."
        ),
    },
    "PixArt-XL": {
        "manual_verified_steps": 12,
        "manual_visual_check": "red apple coherent, brillant, fond blanc",
    },
    "PixArt-Sigma": {
        "manual_verified_steps": 12,
        "manual_visual_check": "red apple on white plate coherent",
    },
    "Sana-1024-MultiLing": {
        "manual_verified_steps": 12,
        "manual_visual_check": "red apple on white plate coherent",
    },
}

for slug, patch in PATCHES.items():
    sd = ROOT / slug
    sj = sd / "stats.json"
    if not sj.exists():
        continue
    s = json.loads(sj.read_text())
    s.update(patch)
    sj.write_text(json.dumps(s, indent=2, ensure_ascii=False))
    md_path = sd / "verdict.md"
    if md_path.exists():
        md = md_path.read_text()
        # Patch verdict line if changed
        if "verdict" in patch:
            for old in ("FAIL", "PASS", "FAIL_HORS_SCOPE"):
                md = md.replace(f"**Verdict agent**: {old}",
                                f"**Verdict agent**: {patch['verdict']}", 1)
        # Append note if not already present
        notes = []
        if "hors_scope_reason" in patch and "Hors-scope" not in md:
            notes.append(f"\n\n## Hors-scope note\n{patch['hors_scope_reason']}\n")
        if "structure_only_reason" in patch and "Structure-only" not in md:
            notes.append(f"\n\n## Structure-only note (Hocine manual review)\n{patch['structure_only_reason']}\n")
        if "manual_visual_check" in patch and "Manual visual" not in md:
            notes.append(f"\n\n## Manual visual check (Hocine review)\n"
                         f"steps={patch['manual_verified_steps']}: {patch['manual_visual_check']}\n")
        if notes:
            md += "".join(notes)
        md_path.write_text(md)
    print(f"{slug:24s} → {patch.get('verdict', 'noted')}")

print("\nReclassification applied.")
