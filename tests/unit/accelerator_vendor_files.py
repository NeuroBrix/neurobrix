"""The accelerators' vendor files (`config/vendors/<vendor>/<arch>.yml` whose device is not the host CPU): what every
"each profile declares X" test reads. A `cpu/` vendor file describes the host CPU as an execution device and holds only
the keys the CPU path reads (cpu/x86_64.yml, cpu/arm64.yml — the key census of 2026-11-01); the accelerator contracts
(shared memory, compute capability, flash, conv layout, fp64 per branch, attention scores) are not its to state."""
from pathlib import Path

import yaml

VENDORS = Path(__file__).resolve().parents[2] / "src" / "neurobrix" / "config" / "vendors"


def accelerator_vendor_files():
    """Sorted paths of every vendor file whose `device_prefix` is not `cpu`; refused when none is found."""
    out = sorted(p for p in VENDORS.glob("*/*.yml")
                 if (yaml.safe_load(p.read_text()) or {}).get("device_prefix") != "cpu")
    assert out, f"no accelerator vendor file under {VENDORS}: every check reading them would be vacuous"
    return out
