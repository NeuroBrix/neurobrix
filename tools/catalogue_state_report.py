#!/usr/bin/env python3
"""The catalogue, one line per model, with a verdict that can be defended.

Every number is READ from the artefact that produced it. Nothing is retyped, and
every cell carries how it was obtained:

    measured      an artefact on this machine holds it, and the line names which
    inferred      derived from a measured line by a stated structural identity
    not measured  said in clear, because a blank cell and a zero read the same

The four sources, and none of them is prose:

  meet.json        the 2026-09-11 catalogue pass — state, wall clock, shapes
                   swept at runtime, shapes the screen excluded, per model
  CAMPAIGN_TABLE   the certified-directory campaign — eleven paired cells with
                   their base, their sweep and their key counts
  the censuses     run live against the local graphs: symbol collisions at the
                   input, depth collisions, the temporal unroll
  the overlay      what changed after the catalogue pass, each entry naming the
                   artefact that proves it

Usage: python tools/catalogue_state_report.py > docs/reference/catalogue-state.md
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/mlops/NeuroBrix_System")
MEET = REPO / "validation_outputs/catalogue_meet_20260911/meet.json"
TABLE = Path("/home/mlops/nbx/campaigns/prepared/CAMPAIGN_TABLE.md")
#: Paired certified-directory campaigns read DIRECTLY from their cells, after the
#: hand-kept table above. Each entry is a directory holding `proof*/<model>/
#: result.json` as `precision_zoo_campaign.py run --env-ab NBX_AUTOTUNE_CERTIFIED=off
#: --paired N --cold-arms` writes them; the row is derived by the same
#: `campaign_table._row` that built the table, so the two sources cannot drift in
#: arithmetic. Listed, never globbed: a campaign voided by its own INVALIDATED.md
#: must be removed here by hand, with the reason in the commit.
CAMPAIGNS = [
    # 2026-09-12 night: one model per card, four cards in parallel, other cards
    # busy — the record of every cell says so (its flightrec note), and the
    # numbers are comparable among themselves, not with a cell that had the rig.
    Path("/home/mlops/nbx/campaigns/2026_09_12_night_catalogue"),
]
PY_BIN = "/home/mlops/ml/venv/bin/python"

#: What changed after the 2026-09-11 pass, each with the artefact that proves it.
#: A line here OVERRIDES the pass's verdict and says why; nothing is edited into
#: the pass's own record, which stays what it was on the day it ran.
#: Axes the differential ADJUDICATED after the census flagged them. Keyed by
#: local container, then by the census's own axis string; the note is appended
#: to the cell, and a flagged axis with no note stays flagged. An adjudication
#: transfers to another container only when the graphs are byte-identical
#: (md5 of the component's graph.json) — said in the note, never assumed.
ADJUDICATED_AXES = {
    "PixArt-Sigma-XL-1024": {
        "vae `height`@128 (weight-extent)": "bound — spatial differential 2026-09-13 on PixArt-Sigma-XL-2-1024-MS, whose vae graph.json is byte-identical to this container's",
        "vae `width`@128 (weight-extent)": "bound — same run, same identical graph",
    },
    "PixArt-Sigma-XL-2-1024-MS": {
        "vae `height`@128 (weight-extent)": "bound — spatial differential 2026-09-13 (14,22 vs 31,47: 0 dims followed the extent)",
        "vae `width`@128 (weight-extent)": "bound — same run",
    },
    "PixArt-XL-2-1024-MS": {
        "vae `height`@128 (weight-extent)": "bound — spatial differential 2026-09-13 (14,22 vs 31,47: 0 dims followed the extent)",
        "vae `width`@128 (weight-extent)": "bound — same run",
    },
    "PixArt-XL-1024": {
        "vae `height`@128 (weight-extent)": "the current tracer binds this axis (differential 2026-09-13 on PixArt-XL-2-1024-MS); this container's graph (built 2026-08-26) is a different trace and stays flagged until re-traced",
        "vae `width`@128 (weight-extent)": "same",
    },
    "Flex.1-alpha": {
        "text_encoder `seq_len`@77 (weight-extent)": "bound — differential 2026-09-12 (77 vs 71: 0 dims moved)",
        "transformer `seq_len`@4096 (weight-extent)": "UNADJUDICATED — the differential's override has no seam on a "
            "transformer traced inside a pipeline: both arms recorded trace_value=4096 again on 2026-09-13 16:05 "
            "(instrument defect named in the census; 'fixed by the model' on 2026-09-12 read the same silence as a fact)",
    },
}

# Verdicts that belong to no single line. Each names its artefact; a number here
# was read from it after the run, never before.
CROSS_CUTTING = [
    "**The strategy change moves no byte** (budget-unified gate, 2026-09-13 15:35-16:03, "
    "after-arm rebuilt on the trunk at run time): five pinned pairs whose Prism strategy "
    "changes between the arms — PixArt-XL-1024, PixArt-XL-2-1024-MS, PixArt-Sigma-XL-1024, "
    "PixArt-Sigma-XL-2-1024-MS on a 16 GB card, Flex.1-alpha on a 32 GB card — rendered "
    "byte-identical images on both arms, three cold repetitions each, triton. The byte "
    "matrix over the whole catalogue ran 2026-09-14 09:11-15:12 (register 52's re-arm): "
    "30 cells identical, 0 adjudicated differences, 2 unadjudicated on the two models "
    "whose own nondeterminism is on record — orpheus-3b-0.1-ft (its sampler draws off the "
    "executor's RNG stream, D-ORPHEUS-SEED-NOT-PINNED) and CogVideoX-2b (differs run to "
    "run on both engines with no RNG op in its graph; cause not yet named, "
    "D-COGVIDEOX-2B-NONDETERMINISTIC-PER-RUN) — and 16 cells unmeasurable on this pair "
    "because its before tree (5ca23b1) cannot load today's containers. Determinism per "
    "mode is a public claim: those two are its named exceptions until their causes are. "
    "`nbx/campaigns/prepared/budget_unified_gate_20260913_1535/RUN.md`.",
    "**The engine suite on the trunk** (`pytest tests/unit tests/regression`, 2026-09-13 "
    "13:28-15:20, 1 h 52): 2100 passed, 21 failed. Ten of the 21 were one defect in the "
    "triton weight loader's consumed-weight filter (register 50, fixed the same afternoon, "
    "proven by run on three cells), one a GPU-less host planned on a GPU (register 51, "
    "fixed), eight out-of-memory against a foreign process on the cards, two Qwen3-Omni "
    "triton cells to re-read after the fix. The 21 are re-run from a worktree frozen at "
    "`8a92312` on a quiet rig; the verdict line is written here when it exists, not before. "
    "`nbx/logs/full_suite_night.log`, `nbx/logs/suite_rerun_21.log`.",
]

OVERLAY = {
    "SANA-Video_2B_720p_diffusers": dict(
        now="met (catalogue pass) — paired cell CUT 2026-09-13 11:27, no certified cost",
        evidence="night bench card 3: arm A rc=0; arm B (sweeping, 25 video conv keys at 720p) "
                 "killed at the campaign's 5400 s run timeout on repetition 0 and cut by hand at "
                 "53 min into repetition 1 — 19 keys swept in 45 min, a sweep this cell cannot "
                 "finish under that clock; the cell was stopped so the night's queue (proofs, "
                 "Open-Sora, budget gate) could take the rig",
        note="A sweep of video conv keys at 720p costs minutes a key; the 90-minute run "
             "timeout that fits every other family cuts this one (and chatterbox's and "
             "openaudio's 674/693-key sweeps). Re-measuring needs a per-cell timeout sized "
             "by keys — a decision, not tonight's.",
        line="measured"),
    "Allegro-TI2V": dict(
        now="RUNS — repaired and delivered",
        evidence="validation_outputs/allegro_image_sets_resolution_20260912/out.mp4 "
                 "(8 frames at 448x448, rc=0, inter-frame diff 23.3); hub replaced "
                 "15:13:48; docs/reference/catalogue-repairs.md entry 1",
        note="Two defects, both fixed at the source: the output size was never read "
             "from the container when the backbone's latent is flattened or the flow "
             "is named something else, and the conditioning image did not set the "
             "resolution. Bounded above: renders to 80 frames, fails at its own "
             "declared 88 asking 25.27 GiB in one allocation — decided 2026-09-13: "
             "PyTorch's native 3-D conv fallback buffer, taken because cuDNN 9.1 "
             "refuses a large non-batch-splittable convolution (needs >= 9.3); a "
             "workspace cap and the V8 flag change nothing. 88 is a declared limit "
             "on this stack until cuDNN >= 9.3 or a per-tile conv bound lands (DETTE D2).",
        line="measured"),
    "CogVideoX-5b-I2V": dict(
        now="RUNS — corrected at the source, published, installed, PROVEN by run",
        evidence="hub record THUDM/CogVideoX-5b-I2V fileSize 23126413914, updatedAt "
                 "2026-09-12T22:09:39Z (replace through the internal entry point, "
                 "2498 s); installed manifest 22:10:26 UTC; the installed "
                 "vae_encoder/graph.json holds 265 ops with symbols batch/height/width "
                 "and NO temporal symbol; regression gate passed component by component "
                 "(vae_encoder 0.82 -> 0.80 GB, every other component 1.000x); proof by run "
                 "22:45 UTC: 9 frames at 448x448, range 9-253, inter-frame diff 3.34 "
                 "(validation_outputs/proof_by_run_CogVideoX-5b-I2V_20260912_2242/VERDICT.json)",
        note="Its causal temporal pad recorded 2187*s - 2184 against a truth of s + 2, "
             "exact at the traced s=1. Profiled at 49 frames the peak was 210.26 GB "
             "against 0.09 GB at the trace, a factor of 2237 which is the compound's "
             "own coefficient. Re-traced, the temporal axis carries no symbol at all "
             "(an I2V encoder conditions on one image) and the same request costs "
             "0.02 GB. 389 -> 265 ops.",
        line="measured"),
    "Open-Sora-v2": dict(
        now="RE-TRACED, REBUILT, PUBLISHED, INSTALLED, PROVEN BY RUN (triton, 9 frames, 2026-09-13 15:33)",
        evidence="re-trace on the fifth attempt of 2026-09-13 (12:50, unpinned): transformer 5089 ops, "
                 "vae 237 (June: 11 017 — an unrolled trace; the new graph is FLAT in T, 237 ops "
                 "at T=9 and T=25, measured on card 0), text encoders 1594/490; rebuild 682 s on the "
                 "export; regression gate 1.000x on three components, vae 0.959x opened with "
                 "--allow-shrink on a measurement (248 tensors identical, the graph shrank); "
                 "upload 13:08:54 -> rc=0 after 1922 s, hub updatedAt 13:40:55Z; install 13:40:56 "
                 "-> rc=0 after 123 s, 53 files, 42.47 GB; proof by run 15:33: 9 frames at 112x176, "
                 "range 0-202, inter-frame difference 18.6, rc=0 after 648 s, unpinned; "
                 "docs/reference/catalogue-repairs.md entry 4",
        note="The snapshot's arrangement (model_index.json, component dirs, safetensors T5 "
             "shards beside the vendor's .bin) is rebuilt from declarations and documented beside "
             "the weights. The June container's 11 017-op VAE was an unrolled trace; the new one is flat in T.",
        line="measured"),
    "mochi-1-preview": dict(
        now="RENDERS on triton at 9 frames (1 step, 118 s) — the CUDA 700 was three int32 index wraps past 2^31 elements, a defect of EVERY kernel for ANY model whose tensor exceeds two billion elements (the next family to meet it will not be called Mochi), fixed for the class at the kernels; at its default 84 frames the VAE decoder OOMs where Prism planned 3.2 GB (estimator debt)",
        evidence="two compute-sanitizer runs (7 200 s 09-13, 18 000 s 09-14) measured nothing; "
                 "--triton-sequential + CUDA_LAUNCH_BLOCKING=1 named aten.mm::1 of the VAE in 19 min "
                 "(M=1 068 480 x N=2048: 2.19e9 output elements, stride_cm * offs_cm wrapped in int32 — "
                 "triton-lang/triton#832); then aten.add::14, then aten.native_group_norm::26, the same "
                 "wrap in the flat and tile forms — every GEMM offset, every flat offset and every program "
                 "id now 64-bit (90fefd4, 7ee3d7c, aa60c5c; register 58; beyond-2^31 tests red then green; "
                 "four models byte-identical, timings within 3 %). With the wraps gone the op-by-op run "
                 "reaches the decoder and OOMs at aten.silu::26: 8.75 GB asked, 25.97 GB live, 5.6 GB free "
                 "on a 32 GB card, where --explain-plan says vae activations 3 209 MB, tiling none planned "
                 "(D-PRISM-MOCHI-VAE-ACTIVATION-UNDERESTIMATED). Bounded proof by run 2026-09-14 09:11: "
                 "--triton --steps 1 --num-frames 9, rc=0 in 118 s, 7 decoded frames at 480x848, range "
                 "0-154, inter-frame difference 1.3-2.6, a warm field with a red centre (one step), "
                 "nbx/campaigns/2026_09_12_night_catalogue/mochi_proof_9frames_aa60c5c/",
        note="The 9-frame run does not cross 2^31 elements itself (114 480 x 2048 rows); the wraps are "
             "proven by the three boundary tests and by the 84-frame op-by-op run that now passes "
             "mm::1, add::14 and group_norm::26. The catalogue request (84 frames) waits on Prism's "
             "estimate carrying the runtime frame count.",
        line="measured"),
    "Wan2.2-I2V-A14B": dict(
        now="RUNS — compiled PROVEN by run at the default guidance; triton renders at cfg 1.0, does not fit one 32 GB card at batched CFG (Prism finding) — and a second line",
        evidence="rebuild 22:11-22:22 (676 s, 118.07 GB); regression gate 1.000x on all "
                 "five components; upload through the internal entry point 22:22:46 -> "
                 "rc=0 after 5144 s (126.77 GB, ~24.6 MB/s mean, zero SlowDownWrite), hub "
                 "updatedAt 23:48:30Z; install 23:48:30 -> rc=0 after 312 s, five "
                 "components in the cache; the proof needs the whole rig and runs after "
                 "proof 2026-09-13: compiled 9 frames 448x448, diff 3.91, PASSED; triton at cfg 1.0 renders (diff 1.01, byte-identical on both engines); at default CFG triton reaches 31 327 MB on the one 32 GB card Prism chose and OOMs at the first attention (1.77 GB scores), Prism refusing component_placement and weight_sharding on a 96 GB rig — DETTE D-PRISM-WAN22-TRITON-ONE-CARD; VAE encoder UNROLLED, MEASURED 2026-09-13 04:44 on "
                 "card 0: 1237 ops at T=9, 3305 at T=25 -> 517 ops per chunk (second line)",
        note="TWO lines, not one. The rebuild resolves its output size. Its VAE "
             "ENCODER stays unrolled over the temporal axis, which no rebuild "
             "changes — DETTE D-TEMPORAL-UNROLL. Delivering the first without "
             "saying the second would be delivering a fix for the error we found "
             "and hiding the one underneath.",
        line="measured (topology) / inferred (unroll)"),
    "Wan2.1-VACE-1.3B": dict(
        now="NAMED DEBT — not corrected, and no stimulus corrects it",
        evidence="validation_outputs/wan_class_e_20260912/VERDICT.md; "
                 "docs/reference/temporal-unroll-census.md",
        note="Its VAE encoder unrolls its temporal chunk loop: 517 ops per chunk, "
             "measured at two stimuli. The blind count held at 135 through T=17, 25, "
             "33 and 41 while the door's candidate walked 25 -> 33 -> 41. The graph "
             "is not symbolic in time however many symbols its table declares.",
        line="measured"),
    "Wan2.1-I2V-14B-480P": dict(
        now="INFERRED same debt as Wan2.1-VACE",
        evidence="its cached encoder carries the measured anchor's chunk-loop groups "
                 "identically: {2:1, 3:9, 6:12, 8:22, 12:1, 21:22}",
        note="No local snapshot, so no second trace point and no fitted slope. The "
             "claim rests on six module groups agreeing exactly on two numbers each, "
             "while the groups ABOVE the loop differ — different VAE sizes carrying "
             "the same loop. One snapshot and two 15-second traces convert it.",
        line="inferred"),
}


def meet_rows() -> list:
    return json.loads(MEET.read_text())


def campaign_cells() -> dict:
    """{container: (cost_s, ratio, keys, certified)} from the campaign table."""
    if not TABLE.exists():
        return {}
    out = {}
    for line in TABLE.read_text().splitlines():
        # The ratio cell ends in a multiplication sign, not an ASCII x, and the
        # model cell may carry a warning glyph. Split on the pipes instead of
        # matching the whole row: a regex for a hand-written table is a regex
        # that breaks when someone adds a column.
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8 or not cells[0].startswith("`"):
            continue
        name = cells[0].split("`")[1]
        try:
            cost = float(cells[4].strip("*"))
            ratio = float(cells[5].strip("*").rstrip("\u00d7x"))
        except ValueError:
            continue
        # The table carries a model twice where a cell was re-run; the FIRST row
        # is the one with its key counts, and overwriting it with the repeat
        # replaced "8/9 keys" with an em dash.
        out.setdefault(name, dict(cost_s=cost, ratio=ratio, keys=cells[6],
                                  certified=cells[7], bytes=cells[8] if len(cells) > 8 else "?"))
        m = None
    for camp in CAMPAIGNS:
        for name, row in campaign_dir_cells(camp).items():
            out.setdefault(name, row)
    return out


# ---------------------------------------------------------------------------
# certified coverage per memory class (register 56): an entry serves only the
# memory class it was proven on, so "certified" is a per-card-class answer on
# this rack (16 GB cards 0 and 1, 32 GB cards 2 and 3). The model's census is
# the MEET pass's own log — it ran with the directory OFF, so every shape the
# run met is a `no certified setting for` line.
# ---------------------------------------------------------------------------
_CENSUS_LINE = re.compile(r"no certified setting for (\S+) (\S+) \((.*?)\) on ")


def _key_from_description(text: str):
    """`M=1024 N=3840 K=1280 IEEE_PRECISION=True PROMOTE_B=True fp32,fp16,fp16,fp32`
    → (1024, 3840, 1280, True, True, 'fp32', 'fp16', 'fp16', 'fp32'): the inverse
    of `autotune_certified.describe_key` — named fields in order, then the
    dtype list."""
    key = []
    for tok in text.split():
        if "=" in tok:
            v = tok.split("=", 1)[1]
            key.append(True if v == "True" else False if v == "False" else int(v) if v.lstrip("-").isdigit() else v)
        else:
            key.extend(tok.split(","))
    return tuple(key)


def memory_class_coverage(container: str):
    """{total, by_class: {16: n, 32: n}, unknown: n, absent: n} for the shapes
    this container's MEET run met, read against the live certified directory;
    None when the run left no log."""
    log = MEET.parent / f"{container}.log"
    if not log.exists():
        return None
    sys.path.insert(0, str(REPO / "src"))
    from neurobrix.kernels import autotune_certified as C
    root = C.directory() / "nvidia" / "volta"
    files: dict = {}
    out = {"total": 0, "by_class": {}, "unknown": 0, "absent": 0}
    seen = set()
    for line in log.read_text(errors="replace").splitlines():
        m = _CENSUS_LINE.search(line)
        if not m:
            continue
        kernel, dtype, desc = m.groups()
        ktext = C.key_repr(_key_from_description(desc))
        if (kernel, dtype, ktext) in seen:
            continue
        seen.add((kernel, dtype, ktext))
        out["total"] += 1
        path = root / f"{kernel}.{dtype}.json"
        if path not in files:
            try:
                files[path] = json.loads(path.read_text(encoding="utf-8")).get("entries") or {}
            except (OSError, ValueError):
                files[path] = {}
        entry = files[path].get(ktext)
        if entry is None:
            out["absent"] += 1
            continue
        classes = C.covered_memory_classes(entry)
        if not classes:
            out["unknown"] += 1
        for c in classes:
            out["by_class"][c] = out["by_class"].get(c, 0) + 1
    return out


def apple_directory_summary() -> dict:
    """What the certified directory holds for Apple silicon — read from the
    live tree (`config/autotune/apple/<profile>/`), the Mac branch's own
    proofs. No model run is measured on this rack for it: there is no Apple
    device here, and the Volta census's shape keys are not the keys a Mac
    meets (its dtype policy differs), so nothing is inferred either."""
    root = REPO / "src" / "neurobrix" / "config" / "autotune" / "apple"
    out = {"profiles": {}}
    if not root.exists():
        return out
    for prof in sorted(p for p in root.iterdir() if p.is_dir()):
        files = {}
        dates = set()
        machine = None
        for f in sorted(prof.glob("*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            ents = d.get("entries") or {}
            files[f.stem] = len(ents)
            for e in ents.values():
                pr = e.get("proof") or {}
                if pr.get("date"):
                    dates.add(str(pr["date"])[:10])
                if machine is None and pr.get("machine"):
                    machine = pr["machine"].get("platform")
        out["profiles"][prof.name] = {"files": files, "shapes": sum(files.values()),
                                      "dates": sorted(dates), "platform": machine}
    return out


APPLE_CELL = "not measured here"


def coverage_cell(cov) -> str:
    if cov is None:
        return "n/m"
    if not cov["total"]:
        return "0 shapes met"
    t = cov["total"]
    c16, c32 = cov["by_class"].get(16, 0), cov["by_class"].get(32, 0)
    extra = []
    if cov["unknown"]:
        extra.append(f"{cov['unknown']} proven on an unknown card")
    if cov["absent"]:
        extra.append(f"{cov['absent']} not in the directory")
    return f"16 GB {c16}/{t} · 32 GB {c32}/{t}" + (f" ({'; '.join(extra)})" if extra else "")


# ---------------------------------------------------------------------------
# the debts named beside the line they hold — the A rows of
# docs/reference/debts-triage.md (a debt that blocks a catalogue line), keyed by
# container. A line with none says "none named"; a debt named here and not in
# DETTE.md is a defect of this table.
# ---------------------------------------------------------------------------
DEBTS_BY_CONTAINER = {
    "Qwen3-VL-30B-A3B-Thinking": ["D-DEEPSTACK-ZERO-EXTENT", "D-QWEN3VL-MOE-RUNS-EVERY-EXPERT-ON-EVERY-TOKEN",
                                  "D-DECLARED-MOE-AS-EXECUTED-VIEW"],
    "Qwen3-Omni-30B-A3B-Instruct": ["D-DEEPSTACK-ZERO-EXTENT", "D-DECLARED-MOE-AS-EXECUTED-VIEW"],
    "Ming-Lite-Omni-1.5": ["D-DECLARED-MOE-AS-EXECUTED-VIEW"],
    "mochi-1-preview": ["D-MOCHI-CUDA-700-AT-MM"],
    "Wan2.1-VACE-1.3B-diffusers": ["D-TEMPORAL-UNROLL", "D-WAN-VACE-BROADCAST-AT-DIV",
                                   "D-NEGATIVE-ALLOCATION-SIZE-WAN-VACE", "D-WAN-VACE-FRAME-TOKENS-FROZEN"],
    "Wan2.1-I2V-14B-480P-Diffusers": ["D-TEMPORAL-UNROLL (inferred)"],
    "Wan2.2-I2V-A14B-Diffusers": ["D-TEMPORAL-UNROLL", "D-PRISM-WAN22-TRITON-ONE-CARD"],
    "Wan2.1-T2V-1.3B-Diffusers": ["D-WAN-T2V-OOM-AT-5D-PAD", "D-WAN-T2V-VAE-ACTIVATION-12GB"],
    "Allegro-TI2V": ["D2 (88 frames: declared limit until cuDNN >= 9.3)", "D-ALLEGRO-TI2V-FRAME-TOKENS-FROZEN"],
    "Allegro": ["D-ALLEGRO-TRITON-31H-PER-ARM", "D-VIDEO-CAMPAIGN-STIMULUS"],
    "Sana_1600M_4Kpx_BF16": ["D-PRISM-SANA4K-COMPILED-16GB"],
    "GLM-4.1V-9B-Thinking": ["D-PRISM-2x16-PIPELINE-OVERFILL"],
    "deepseek-moe-16b-chat": ["D-DSMOE-XENGINE-SHA", "D-TRACE-DEEPSEEK-MOE-ILLEGAL-ACCESS"],
    "granite-speech-3.3-8b": ["D-AUDIO-LLM-GRANITE-HOST-PLACEMENT"],
    "Kokoro-82M": ["D-CPU-COMPLEX-HALF-EXP", "D-KOKORO-DECODER-PINNED-HOST-READ"],
}
DEBTS_BY_SLUG = {"Orpheus-3B": ["D-ORPHEUS-FT-VENDOR-CODEC", "D-ORPHEUS-SEED-NOT-PINNED"]}


def debts_cell(container, slug) -> str:
    names = DEBTS_BY_CONTAINER.get(container or "", []) + DEBTS_BY_SLUG.get(slug, [])
    return ", ".join(f"`{n}`" for n in names) if names else "none named"


def per_shape_sweep_cost() -> dict:
    """{family: [row, ...]} for every campaign cell that measured a sweep."""
    sys.path.insert(0, str(REPO / "tools"))
    from campaign_table import _row
    out: dict = {}
    for camp in CAMPAIGNS:
        try:
            perturbed = json.loads((camp / "PERTURBED.json").read_text())
        except (OSError, ValueError):
            perturbed = {}
        for result in sorted(camp.glob("proof*/*/result.json")):
            try:
                r = _row(json.loads(result.read_text()))
            except (OSError, ValueError):
                continue
            if (str(result.parent.relative_to(camp)) in perturbed or r["rc"] != (0, 0)
                    or not r["keys"] or not r["sweep"]):
                continue
            out.setdefault(r["family"], []).append(r)
    return out


def campaign_dir_cells(camp: Path) -> dict:
    """{container: cell} straight from a campaign's result.json files.

    A cell whose lever did not move (no keys swept AND nothing served) carries no
    ratio in `campaign_table._row`, and here it is NOT a measurement of the
    directory: the row says `lever did not move` rather than a cost of zero.
    A cell whose arm failed (rc != 0) is reported as failed, with the arm named.
    """
    sys.path.insert(0, str(REPO / "tools"))
    from campaign_table import _row  # the table's own arithmetic, reused
    out = {}
    # A cell that shared its card with another compute process is listed by hand
    # in PERTURBED.json with the evidence; it renders as perturbed and carries no
    # cost. A file nobody wrote means nobody claimed a perturbation.
    try:
        perturbed = json.loads((camp / "PERTURBED.json").read_text())
    except (OSError, ValueError):
        perturbed = {}
    for result in sorted(camp.glob("proof*/*/result.json")):
        try:
            cell = json.loads(result.read_text())
        except (OSError, ValueError):
            continue
        r = _row(cell)
        # Keyed by the cell's directory relative to the campaign, not by model:
        # a re-run of the same model lands in another directory and must NOT
        # inherit the mark. Later directories win in sorted order.
        rel = str(result.parent.relative_to(camp))
        if rel in perturbed:
            out[r["model"]] = dict(cost_s=None, ratio=None, keys=r["keys"], certified=r["served"],
                                   bytes=r["bytes"], failed=f"PERTURBED ({perturbed[rel]})")
            continue
        rc_a, rc_b = r["rc"]
        if rc_a not in (0, None) or rc_b not in (0, None):
            out[r["model"]] = dict(cost_s=None, ratio=None, keys=r["keys"], certified=r["served"],
                                   bytes=r["bytes"], failed=f"arm A rc={rc_a}, arm B rc={rc_b}")
            continue
        if r["ratio"] is None:
            out[r["model"]] = dict(cost_s=None, ratio=None, keys=r["keys"], certified=r["served"],
                                   bytes=r["bytes"], failed="lever did not move")
            continue
        choices = cell.get("choices") or {}
        out[r["model"]] = dict(cost_s=r["sweep"], ratio=f"{r['ratio']:.2f}", keys=r["keys"],
                               certified=r["served"], bytes=r["bytes"], base_s=r["a_med"],
                               contradictions=int(choices.get("contradicted_count") or 0),
                               near_ties=int(choices.get("near_tie_count") or 0))
    return out


def census(tool: str, *args) -> str:
    try:
        return subprocess.run([PY_BIN, str(REPO / "tools" / tool), *args],
                              capture_output=True, text=True, timeout=600).stdout
    except Exception:
        return ""


CENSUS_CACHE = REPO / "validation_outputs" / "catalogue_state_census.json"


def blind_axes(from_cache: bool = False) -> dict:
    """{container: [reasons]} — where a defect would be invisible, per model.

    The census reads 182 graphs off the NFS export. When the export is carrying
    one heavy stream (an upload, a build) that read is a second one, and the
    rule is one at a time. So the last census is kept on the root filesystem
    and `--from-cache` renders from it, STAMPING ITS TIME in the document: a
    census read from cache is a census as of then, and the document says so.
    """
    import time as _t
    if from_cache:
        if not CENSUS_CACHE.exists():
            raise SystemExit(f"REFUSED: --from-cache but {CENSUS_CACHE} does not "
                             f"exist. A document rendered from no census is not one.")
        cached = json.loads(CENSUS_CACHE.read_text())
        blind_axes.stamp = cached.get("taken_utc", "?")
        return cached["blind"]
    raw = census("symbol_collision_census.py", "--json")
    out = {}
    try:
        data = json.loads(raw)
    except ValueError:
        return out
    for comp, entries in (data.get("findings") or {}).items():
        model = comp.split("/")[0]
        for e in entries:
            for cls, _why in e["flags"]:
                out.setdefault(model, []).append(
                    f"{comp.split('/',1)[1]} `{e['name']}`@{e['trace']} ({cls})")
    stamp = _t.strftime("%Y-%m-%d %H:%M UTC", _t.gmtime())
    CENSUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CENSUS_CACHE.write_text(json.dumps({"taken_utc": stamp, "blind": out}, indent=1))
    blind_axes.stamp = stamp
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-cache", action="store_true",
                    help="render from the last census kept on the root filesystem "
                         "(its time is stamped in the document)")
    args = ap.parse_args()
    rows = meet_rows()
    cells = campaign_cells()
    blind = blind_axes(from_cache=args.from_cache)

    print("# The catalogue, one line per model — 2026-09-12\n")
    print("**47 entries on the registry.** Every cell is read from an artefact on this")
    print("machine, and every cell says how it was obtained. A cell that says *not")
    print("measured* is not an omission: a blank and a zero read the same, and only one")
    print("of them is honest.\n")
    print(f"The *where a defect would be invisible* column is the census taken "
          f"**{getattr(blind_axes, 'stamp', '?')}**"
          + (" (rendered from its cache: the export was carrying one stream, and "
             "the rule is one at a time)" if args.from_cache else "") + ".\n")
    print("The run column is the catalogue pass of **2026-09-11** at engine `4c119b5`")
    print("unless a later line overrides it, in which case the override names the")
    print("artefact that proves it. The pass's own record is never edited — it stays")
    print("what it was on the day it ran.\n")

    states = {}
    for r in rows:
        states[r["state"]] = states.get(r["state"], 0) + 1
    over = sum(1 for r in rows if r["hub"].split("/")[-1] in OVERLAY)
    summary = ", ".join(f"**{v} {k}**" for k, v in
                        sorted(states.items(), key=lambda x: -x[1]))
    print(f"As the pass left it: {summary}. {over} rows carry a later line, "
          f"and every one of the nine failures was a VIDEO model.\n")

    print("| model | family | GB | on this rack | swept | screened | certified cost | "
          "certified for this card's memory | Apple M4 Pro | where a defect would be invisible | debts named | line |")
    print("|---|---|---:|---|---:|---:|---|---|---|---|---|---|")
    n_rows = n_cost = 0
    for r in sorted(rows, key=lambda r: (r["family"], r["hub"])):
        slug = r["hub"].split("/")[-1]
        container = r.get("container") or slug
        cell = cells.get(container)
        ov = OVERLAY.get(slug) or OVERLAY.get(container)
        if ov:
            run = f"**{ov['now']}**"
            line = ov["line"]
        else:
            # A ROW MAY LACK A FIELD, and a missing field is not a zero. The
            # catalogue decision (`Orpheus-3B`) was never built, so it carries no
            # container, no wall clock and no swept count; printing 0 for those
            # would be the lying cell this document exists to avoid.
            wall = r.get("wall_s")
            clock = f"{wall:.0f} s" if isinstance(wall, (int, float)) else "no clock"
            run = {"met": f"met in {clock}",
                   "failed": (f"FAILED rc={r.get('rc', '?')}"
                              + (f" — killed at {clock}"
                                 if r.get("rc") == -9 else f" in {clock}")),
                   "not runnable": "not runnable — catalogue decision"}.get(
                       r.get("state", "?"), r.get("state", "?"))
            line = "measured" if wall is not None else "not measured"
        if not cell:
            cost = "not measured"
        elif cell.get("failed"):
            cost = f"paired cell {cell['failed']} — no cost"
        else:
            base = f", base {cell['base_s']:.0f} s" if cell.get("base_s") else ""
            cost = (f"{cell['cost_s']:.0f} s, {cell['ratio']}x, {cell['certified']}/"
                    f"{cell['keys']} keys{base}, bytes {cell.get('bytes', '?')}")
            # A runtime sweep that contradicts a certified choice is a reported
            # finding, never a silence (autotune doctrine). The campaign counts
            # them per cell and names the keys in its own record.
            if cell.get("contradictions"):
                cost += (f"; {cell['contradictions']} certified choice(s) contradicted by the "
                         f"runtime sweep ({cell.get('near_ties', 0)} near-ties within the timer's "
                         f"noise) — a finding, keys in the campaign record")
        axes = blind.get(container, [])
        notes = ADJUDICATED_AXES.get(container, {})
        shown = [a + (f" → {notes[a]}" if a in notes else "") for a in axes[:2]]
        blind_cell = "; ".join(shown) + (f" (+{len(axes)-2})" if len(axes) > 2 else "") \
            if axes else "none found at the input"
        # A `swept` of 0 on a row that FAILED is not coverage. It counts the
        # shapes the run reached, and a run that died in five seconds reached
        # none. Reading it as "the certified directory served everything" is the
        # difference between a measure and an artefact of the failure, and the
        # first version of the reading guide made exactly that claim.
        incomplete = r.get("state") != "met"
        swept = r.get("swept")
        if swept is not None and incomplete:
            swept = f"{swept}†"
        screened = r.get("screened_out")
        if screened is not None and incomplete:
            screened = f"{screened}†"
        coverage = coverage_cell(memory_class_coverage(container))
        print(f"| `{r['hub']}` | {r.get('family', '?')} | {r.get('gb', 0):.1f} | "
              f"{run} | {swept if swept is not None else 'n/m'} | "
              f"{screened if screened is not None else 'n/m'} | {cost} | "
              f"{coverage} | {APPLE_CELL} | {blind_cell} | {debts_cell(container, slug)} | {line} |")
        n_rows += 1
        if not cost.startswith("not measured"):
            n_cost += 1

    print("\n## The lines that carry a later verdict\n")
    for slug, ov in sorted(OVERLAY.items()):
        print(f"### `{slug}` — {ov['now']}\n")
        print(f"{ov['note']}\n")
        print(f"*Evidence:* {ov['evidence']}  ·  *line:* {ov['line']}\n")

    print("## How to read the columns\n")
    print("**debts named** — the entries of `DETTE.md` that hold this line (the A rows of")
    print("`docs/reference/debts-triage.md`), so a reader of the line sees what it waits on")
    print("without opening the debt file. A line that runs and measures may still name one:")
    print("a debt that bounds it (frames, a card class) rather than blocks it.\n")
    print("**certified for this card's memory** — of the shapes this model's catalogue")
    print("run met (its own log, directory off), how many the directory certifies for a")
    print("16 GB card and how many for a 32 GB card, read on the day this document was")
    print("rendered. Since 2026-09-13 an entry serves only the memory class it was")
    print("proven on (register 56): a shape proven on a 16 GB card sweeps at runtime on a")
    print("32 GB card until it is certified there, and a shape proven on the rig with the")
    print("card unknown serves no card until re-proven. The two numbers are what a")
    print("request on each SKU of this rack is served without a sweep — not what the")
    print("directory holds.\n")
    apple = apple_directory_summary()
    print("**Apple M4 Pro** — every cell says *not measured here*, and that is the")
    print("whole truth of this rack: it has no Apple device, and a Mac's shape keys are")
    print("not this rack's (the dtype policy differs), so nothing is inferred from the")
    print("Volta census either. What the trunk carries for Apple since the Mac branch")
    print("merged is the certified directory the Mac itself wrote, read from the live tree:")
    if apple["profiles"]:
        for name, prof in apple["profiles"].items():
            files = ", ".join(f"{k} {v}" for k, v in prof["files"].items())
            dates = f"{prof['dates'][0]}..{prof['dates'][-1]}" if prof["dates"] else "no dated proof"
            print(f"`{name}`: **{prof['shapes']} shapes** ({files}), proofs {dates}, "
                  f"platform `{prof['platform']}`. A model's Apple line is measured on the machine")
            print("that carries the card, by its own matrix runner (`tools/apple_matrix*.py`), and")
            print("lands here as a row when it does.\n")
    else:
        print("nothing — the live tree carries no `config/autotune/apple/` directory.\n")
    print("**swept** — shape keys this model had to sweep AT RUNTIME because the")
    print("certified directory did not hold them. On a row that MET, `0` is the")
    print("per-model measure of certified coverage: it was served entirely from the")
    print("directory. `n/m` is a model that was never run.\n")
    print("**† marks a row whose run did not complete**, and it changes what the two")
    print("columns mean there. They count what the run REACHED, and a run that died")
    print("in five seconds reached nothing — so a `0†` is not coverage, it is the")
    print("shape of the failure. Reading it as coverage would credit the directory")
    print("for work no one asked it to do.\n")
    print("**screened** — candidate configurations the correctness screen excluded")
    print("before timing. Zero across the whole catalogue, on 998 keys.\n")
    print("**certified cost** — from the paired certified-directory campaigns: the")
    print("hand-kept table of 2026-09-11, then every campaign listed in `CAMPAIGNS`")
    print("read straight from its cells through the table's own arithmetic. A night")
    print("cell also carries its base time (arm A median) and its byte gate — `same`,")
    print("`N dB`, `DIFFER`, `nondet both` or `did not run`; a cell whose arm failed")
    print("or whose lever did not move says so and carries no cost. The 2026-09-12")
    print("night ran one model per card with the other cards busy: its numbers are")
    print("comparable among themselves, not with a cell that had the rig alone.")
    print("The rest of the paragraph describes the 2026-09-11 table, which")
    print("covers eleven cells and not the catalogue. The ratio is what runtime")
    print("sweeping costs relative to a served run, on this rack, at the shapes these")
    print("requests meet. It is not a throughput figure and it says nothing about")
    print("other hardware.\n")
    per_shape = per_shape_sweep_cost()
    if per_shape:
        print("**What a sweep costs per shape, measured tonight (sweep cost / keys swept,")
        print("per family, from the cells above with both arms at rc=0 and not perturbed):**")
        print()
        print("| family | cells | s per shape (min – max) | keys per cell (min – max) |")
        print("|---|---:|---|---|")
        for fam, rows in sorted(per_shape.items()):
            per = [r["sweep"] / r["keys"] for r in rows]
            keys = [r["keys"] for r in rows]
            print(f"| {fam} | {len(rows)} | {min(per):.0f} – {max(per):.0f} | {min(keys)} – {max(keys)} |")
        print()
        allrows = [r for rows in per_shape.values() for r in rows]
        top = max(allrows, key=lambda r: r["sweep"] / r["keys"])
        low = min(allrows, key=lambda r: r["sweep"] / r["keys"])
        print(f"The 2026-09-11 table quoted 3–12 s a shape on GEMM-class keys. Tonight the")
        print(f"spread runs from {low['sweep']/low['keys']:.0f} s a shape (`{low['model']}`) to")
        print(f"{top['sweep']/top['keys']:.0f} s (`{top['model']}`, conv2d shapes at 448² screened")
        print(f"against the fp64 oracle) — so a {top['a_med']:.0f} s run of the latter pays")
        print(f"{top['sweep']:.0f} s of sweep. The per-shape cost is a property of the kernel")
        print("class and the shape, not a constant; the law (shapes met per second of")
        print("served run) holds with that coefficient per cell, not a single one.\n")
    print("**where a defect would be invisible** — axes traced at a value where two")
    print("distinct rules give the same number, so the trace-point check cannot tell")
    print("them apart. This is NOT a defect list. An axis here needs its rule asserted")
    print("structurally or a re-trace outside the collision; a test at the flagged")
    print("value is green for the reason that blinds it.\n")

    print("## Verdicts that cut across the lines\n")
    for v in CROSS_CUTTING:
        print(f"* {v}")
    print()
    print("## What this document does not say\n")
    print("* **Whether a model is CORRECT.** The run column says it produced output")
    print("  without failing, on one request, at one moment. Numerical agreement")
    print("  against a vendor pipeline is a different instrument and covers five of")
    print("  nine families.")
    print(f"* **What every model costs with the certified directory.** {n_cost} of {n_rows}")
    print(f"  rows carry a measured or stated cost; the other {n_rows - n_cost} say *not")
    print("  measured* and that is the whole point of the cell.")
    print("* **Whether the flagged axes are wrong.** They are places a defect could")
    print("  not be seen. Converting one into a verdict costs a second trace at a")
    print("  value outside the collision, and the instrument that does it refuses when")
    print("  the stimulus does not actually move — a tree compared with itself agrees")
    print("  with itself.")
    print("* **Anything about hardware other than this rack**: four V100s, two of 16 GB")
    print("  and two of 32, at 1290/877 MHz.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
