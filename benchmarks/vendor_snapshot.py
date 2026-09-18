"""Vendor tools read the snapshot in place. They never download a second copy.

A snapshot is downloaded ONCE, into the configured snapshots root, and the build
toolchain and the vendor tools both read it THERE. Until 2026-09-17 the profilers
did this instead:

    local_path = Path("/home/mlops/hf_snapshots/Sana_1600M_1024px_MultiLing")
    model_id = str(local_path) if local_path.exists() else "Efficient-Large-Model/..."

— a literal path, and "otherwise try Hub", which downloads a second copy into
`~/.cache/huggingface/hub` whenever the name on disk does not match the one in
the code. That cache held 4.0 G across 25 entries when this was written, and the
local disk had reached 0 MB the same day.

So: the root comes from `config/system.yml`, a missing snapshot REFUSES by name
with the directory it looked in, and there is no Hub branch to fall through to.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

SYSTEM_YML = Path(__file__).resolve().parent.parent / "src" / "neurobrix" / "config" / "system.yml"


class SnapshotNotPresent(Exception):
    """A vendor snapshot is not in the snapshots root, and we do not fetch it here."""


def snapshots_root() -> Path:
    """The configured snapshots root. `NEUROBRIX_HF_SNAPSHOTS` overrides explicitly."""
    env = os.environ.get("NEUROBRIX_HF_SNAPSHOTS")
    if env:
        return Path(env)
    cfg = yaml.safe_load(SYSTEM_YML.read_text()) or {}
    root = (cfg.get("paths") or {}).get("snapshots")
    if not root:
        raise SnapshotNotPresent(
            f"no snapshots root is configured: looked at $NEUROBRIX_HF_SNAPSHOTS and "
            f"{SYSTEM_YML} under `paths.snapshots`. There is deliberately no default.")
    return Path(root)


def vendor_snapshot(*names: str) -> Path:
    """The first of `names` present under the snapshots root, or a refusal.

    Several names because a vendor directory is not always spelled like the hub
    id (`Sana_1600M_1024px_MultiLing` on disk, `..._BF16` on the hub).
    """
    root = snapshots_root()
    for n in names:
        p = root / n
        if p.is_dir():
            return p
    raise SnapshotNotPresent(
        f"none of {list(names)} is in {root}. Download it THERE once — this tool "
        f"reads snapshots in place and will not fetch a second copy into the "
        f"Hugging Face hub cache, which must stay empty.")
