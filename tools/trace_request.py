#!/usr/bin/env python3
"""The request a tool runs a model at, DERIVED from the current container's trace — one place.

Two tools must ask a model for the SAME request: the regression matrix, whose cells are the
verification, and the certification census, whose keys are what the verification is served
from. On 2026-09-28 they did not. Sana_1600M_1024px_MultiLing was retraced (its trace moved to
960x1088); the census took its request from a static table written from the OLD container
(768x1024) while the matrix derived 704x1088 from the new one. 72 keys were certified for a
request the verification never makes, and the zero-miss verification met 42 uncertified keys.
The rule (supervisor, 2026-09-28, restating the owner): after any retrace, a model's census is
re-run on the NEW container at the request derived from the new trace — never a stale table.

So the derivation lives here and both tools import it:

* `off_trace_size(model, family)` — for an image or video request, the container's own size
  (`resolution.container_size`, the executor's and the plan's one authority) with the height
  taken to three quarters on the family's lattice (64 pixels for an image, 32 for a video) and
  the width kept: a non-square request away from the trace, the one class that catches a
  swapped or frozen spatial axis. None for the families whose judged request is already away
  from its trace extents (a prompt is not 23 tokens, a clip is not the trace clip).
* `derived_request(model, family)` — the family's judged request (`precision_zoo_campaign.
  request_args`: its calibration section, its media, its bound) at that size.
* `requested_size(request)` — the (height, width) a request asks for, as the CLI reads it.

The container is read through `core.paths.cache_dir()` at call time — the same door the engine
run reads — so a derivation and the run it describes can never look at two different caches.

Writing a request table (for a tool that takes one, e.g. `certified_census.py --requests-json`)
from the containers as they are NOW:

    python tools/trace_request.py --models A,B --out requests.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import precision_zoo_campaign as Z  # noqa: E402  the judged request (request_args), the family

#: The spatial lattice an off-trace size is taken on, per family whose request names a size.
LATTICE = {"image": 64, "video": 32}
#: The request flags that carry a spatial size — what a request table must agree on.
SIZE_FLAGS = ("--height", "--width")


def off_trace_size(model: str, family: str) -> Optional[Tuple[int, int]]:
    """(height, width) for an image or video request: the container's own size, height at three
    quarters on the family's lattice, width kept. None for the families whose judged request is
    already away from the trace, or when the container states no size (said in the row)."""
    if family not in LATTICE:
        return None
    from neurobrix.core.paths import cache_dir
    from neurobrix.core.runtime.loader import NBXRuntimeLoader
    from neurobrix.core.runtime.resolution.container_size import container_output_size
    pkg = NBXRuntimeLoader().load(str(cache_dir() / model))
    size = container_output_size(pkg.manifest, pkg.defaults,
                                 pkg.topology.get("components", {}) or {}, pkg.components)
    if size is None:
        # The container states no size: the engine then renders at the family's own default
        # (executor: "a family constant is the last resort"), which is what it was traced at.
        from neurobrix.core.config import get_family_config
        fam_defaults = get_family_config(family).get("defaults") or {}
        if "height" not in fam_defaults or "width" not in fam_defaults:
            return None
        size = (fam_defaults["height"], fam_defaults["width"])
    h, w = (int(v) for v in size)
    step = LATTICE[family]
    h2 = max(step, (h * 3 // 4) // step * step)
    return (h2, w) if h2 != h else (max(step, h - step), w)


def derived_request(model: str, family: Optional[str] = None) -> list:
    """The request a model is run at: the family's own judged request at the container's
    off-trace size (no size flag for a family that has none)."""
    family = family or Z.family_of(model)
    req = Z.request_args(model, family, [])
    size = off_trace_size(model, family)
    if size is not None:
        req = req + ["--height", str(size[0]), "--width", str(size[1])]
    return req


def requested_size(request: list) -> Optional[Tuple[Optional[int], Optional[int]]]:
    """(height, width) a request asks for — the LAST occurrence of each flag, as argparse reads
    it; a flag the request does not carry is None; None when it carries neither."""
    got = {}
    for i, a in enumerate(request):
        if a in SIZE_FLAGS:
            if i + 1 >= len(request):
                raise ValueError(f"request ends on {a} with no value: {request}")
            got[a] = int(request[i + 1])
    if not got:
        return None
    return got.get("--height"), got.get("--width")


def size_text(size) -> str:
    """A size for a message: `704x1088`, `none` when the request names no size."""
    if size is None:
        return "none"
    return "x".join("?" if v is None else str(v) for v in size)


def main() -> int:
    ap = argparse.ArgumentParser(description="Write the request table derived from the containers as they are now.")
    ap.add_argument("--models", required=True, help="comma-separated container names")
    ap.add_argument("--out", required=True, help='requests.json: {"<model>": [[flags...]]}')
    a = ap.parse_args()
    table = {m: [derived_request(m)] for m in (x.strip() for x in a.models.split(",")) if m}
    Path(a.out).write_text(json.dumps(table, indent=1))
    for m, (req,) in table.items():
        print(f"[trace_request] {m}: {size_text(requested_size(req))}  {' '.join(req)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
