"""`neurobrix calibrate` — measure a model's precision calibration record.

One request on the conservative reference path (every matmul stored fp32
on fp16 hardware — the engine's default for an uncalibrated component) with
the DtypeEngine's per-op magnitude census installed; at the end, one record
per component (core/dtype/calibration.py) in the engine store, from which
every later run derives its fp32 islands for the compute dtype at hand.
How a record travels with a distributed artifact is the owner's format
decision (DETTE D-PRECISION-CONTRACT-DEPLOYMENT-SPLIT); the store serves.

The stimulus is the request itself: the arguments of `run`, completed for a
family that needs no media input by `calibration:` in
config/families/<family>.yml (prompt, steps, seed…). A family that needs
media (speech, vision) takes it from the command line, as `run` does.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

# The request fields that describe the stimulus in the record.
_STIMULUS_FIELDS = ("prompt", "audio", "input_image", "input_video", "reference_audio",
                    "reference_image", "steps", "cfg", "height", "width", "seed",
                    "max_tokens", "num_frames", "fps", "mode", "speaker", "temperature")
_FILE_FIELDS = ("audio", "input_image", "input_video", "reference_audio", "reference_image")


def _identity_of(model: str):
    """(family, manifest model_name) of a cached model — the record key is the
    manifest's name, never the directory name (hub-slug installs differ)."""
    from neurobrix.cli.commands.run import find_model
    from neurobrix.nbx import NBXContainer
    manifest = NBXContainer.load(str(find_model(model))).get_manifest() or {}
    family = manifest.get("family")
    if not family:
        raise RuntimeError(f"ZERO FALLBACK: 'family' missing in manifest for model '{model}'")
    return family, manifest.get("model_name") or model


def _stimulus_of(args) -> Dict[str, Any]:
    """The request fields that describe the stimulus; file inputs by basename
    only, so a record never carries a local path."""
    out = {}
    for k in _STIMULUS_FIELDS:
        v = getattr(args, k, None)
        if v is None:
            continue
        out[k] = os.path.basename(str(v)) if k in _FILE_FIELDS else v
    return out


def _apply_family_stimulus(args, family: str) -> Dict[str, Any]:
    """Fill the request from the family's `calibration:` section where the
    command line left a field unset; returns the stimulus actually used."""
    from neurobrix.core.config.loader import get_family_config
    stim = dict((get_family_config(family) or {}).get("calibration") or {})
    for key, value in stim.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return _stimulus_of(args)


def cmd_calibrate(args) -> int:
    from neurobrix.core.dtype import calibration as cal
    from neurobrix.core.config.loader import get_precision_calibration_policy
    from neurobrix.core.runtime.precision_contract import FLAG_ENV
    from neurobrix.serving.client import DaemonClient

    if args.model is None:
        print("ERROR: --model is required for calibrate.")
        return 2
    if DaemonClient.is_running():
        print("ERROR: a serving daemon is running — the census must run in this process. "
              "Stop it first (neurobrix stop).")
        return 2
    if getattr(args, "triton", False) or getattr(args, "triton_sequential", False):
        print("ERROR: the census runs on the compiled reference path (the numerical oracle); "
              "the record it writes serves every engine. Drop --triton / --triton-sequential.")
        return 2
    family, model_name = _identity_of(args.model)
    stimulus = _apply_family_stimulus(args, family)
    policy = get_precision_calibration_policy()

    # The reference path, whatever record exists today; the run's output is
    # kept beside the records (R29: an inspectable artefact of the census).
    os.environ[FLAG_ENV] = "0"
    out_dir = cal.STORE_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "output", None) is None:
        args.output = str(out_dir / "calibration_output")
        from neurobrix.core.runtime.output_dispatch import resolve_output_path
        args.output = resolve_output_path(args.output, args.model, family, getattr(args, "mode", None))

    print("=" * 70)
    print(f"NeuroBrix Calibrate — precision census on the {policy['reference']} path")
    print(f"   stimulus: {stimulus}")
    print("=" * 70)
    session = cal.begin_calibration()
    try:
        from neurobrix.cli.commands.run import cmd_run
        try:
            rc = cmd_run(args)
        except SystemExit as e:          # the run command ends some families with an exit
            rc = int(e.code or 0)
    finally:
        cal.end_calibration()
    if rc:
        print(f"[calibrate] the reference run failed ({rc}); no record written")
        return int(rc)

    written = []
    for component, census in session.items():
        if census.dag is None:
            print(f"[calibrate] {component}: census never bound to a graph — skipped")
            continue
        record = cal.CalibrationRecord.build(
            model_name, component, census.dag, census.finalize(),
            stimulus=stimulus, passes=census.passes, reference=str(policy["reference"]),
            non_finite=census.non_finite_ops(), graph_signature=census.signature)
        path = cal.store_path(model_name, component)
        record.save(path)
        top = max(record.max_abs.values()) if record.max_abs else 0.0
        print(f"[calibrate] {component}: {len(record.max_abs)} op(s) over {record.passes} pass(es), "
              f"largest finite magnitude {top:.4g}, {len(record.non_finite)} op(s) with a "
              f"non-finite value on the reference → {path}")
        written.append(path)
    if not written:
        print("[calibrate] no component was measured — nothing written")
        return 1
    return 0
