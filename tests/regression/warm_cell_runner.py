"""One warm serve cell in a fresh process.

Invoked by tests/regression/test_serve_warm.py as a subprocess — one
model per process, which is BOTH the production serve shape (one
daemon per model) and the isolation the in-process sweep needs:
sequential load/unload cycles in a single process accumulate the
triton live set even after `unload()` + gc (module-global holders
retain NBXTensor refs — named chantier P-SERVE-UNLOAD-LIVE-SET,
warm-sweep finding 2026-08-26; the doctrine gc.collect in unload()
is necessary but not sufficient).

argv[1] = JSON {model, mode, gen_kwargs, verify}. Prints
WARM-CELL-OK on success; any failure prints the reason and exits
nonzero. Output artifacts go to a TemporaryDirectory.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def main() -> int:
    spec = json.loads(sys.argv[1])
    model = spec["model"]
    mode = spec["mode"]
    gen_kwargs = spec["gen_kwargs"]
    verify = spec["verify"]

    from neurobrix.core.prism.autodetect import get_or_create_default_profile
    from neurobrix.serving.engine import InferenceEngine

    # Optional pinned form: mask + MATCHING single-GPU hardware profile
    # together (the bench closure-config pattern — autodetect
    # enumerates via nvidia-smi and is blind to CUDA_VISIBLE_DEVICES:
    # D-AUTODETECT-VISIBLE-MASK).
    hardware = spec.get("hardware") or get_or_create_default_profile()
    engine = InferenceEngine(model, hardware, mode=mode)
    engine.load()
    result = engine.generate(**gen_kwargs)

    tag = f"{model}-{mode}"
    if verify == "text":
        text = result.get("text")
        assert isinstance(text, str) and text.strip(), (
            f"{tag}: empty/missing text in warm result — keys "
            f"{sorted(result)}")
    else:
        kind = verify.split(":", 1)[1]
        assert "outputs" in result, (
            f"{tag}: no outputs in warm result — keys {sorted(result)}")
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / f"warm_{tag}.{kind}"
            saved = engine.save_output(result["outputs"], str(out_path),
                                       mode=gen_kwargs.get("mode"))
            p = Path(saved)
            assert p.exists() and p.stat().st_size > 1024, (
                f"{tag}: saved output missing or trivial: {saved}")
            if kind == "png":
                from PIL import Image
                im = Image.open(p)
                assert im.mode == "RGB" and min(im.size) >= 256, (
                    f"{tag}: implausible image output {im.size} {im.mode}")

    print("WARM-CELL-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
