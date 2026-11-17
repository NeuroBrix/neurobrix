"""The accelerators' vendor files (`config/vendors/<vendor>/<arch>.yml` whose device is not the host CPU): what every
"each profile declares X" test reads. A `cpu/` vendor file describes the host CPU as an execution device and holds only
the keys the CPU path reads (cpu/x86_64.yml, cpu/arm64.yml — the key census of 2026-11-01); the accelerator contracts
(shared memory, compute capability, flash, conv layout, fp64 per branch, attention scores) are not its to state."""
from pathlib import Path

import yaml

VENDORS = Path(__file__).resolve().parents[2] / "src" / "neurobrix" / "config" / "vendors"


def documentation_sourced_unmeasured(path) -> bool:
    """A vendor file that says it is DOCUMENTATION-SOURCED AND UNMEASURED (written with no such card, every value read
    from the vendor's documentation: nvidia/ada.yml, GitHub issue 6) and states no Triton dead-tensor queue
    (`memory.deferred_drain_cliff_bytes`), so a Triton plan on it is refused by name. Such a file carries none of the
    Triton branch's MEASURED tuning values (the owner's rule of 2026-11-17: never copied from a neighbour)."""
    text = Path(path).read_text()
    memory = (yaml.safe_load(text) or {}).get("memory") or {}
    return "DOCUMENTATION-SOURCED AND UNMEASURED" in text and "deferred_drain_cliff_bytes" not in memory


def accelerator_vendor_files():
    """Sorted paths of every vendor file whose `device_prefix` is not `cpu`; refused when none is found."""
    out = sorted(p for p in VENDORS.glob("*/*.yml")
                 if (yaml.safe_load(p.read_text()) or {}).get("device_prefix") != "cpu")
    assert out, f"no accelerator vendor file under {VENDORS}: every check reading them would be vacuous"
    return out
