"""The census's coverage line counts what the certified directory serves ON ITS MEMORY CLASS.

On 2026-09-26 the census printed "directory nvidia/volta: 0 served, 743 to certify" over a
directory of thousands of entries: it asked `lookup`, which resolves the vendor profile from the
driver, in a process that sees no card by design, and a bare `except: continue` turned the
failure into "not served". It also asked `any_class`, which would count an entry proven on a
32 GB card as served to a 16 GB census.

Here, with no card visible: a real entry of the directory is served on the class its proof
names and not on another class. On the old reader both fail (0 served, and any_class).
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

CHILD = textwrap.dedent('''
    import json, sys
    from pathlib import Path
    sys.path.insert(0, "tools")
    import certified_census as CC
    from neurobrix.kernels import autotune_certified as C
    root = Path("src/neurobrix/config/autotune/nvidia/volta")
    picked = None
    for p in sorted(root.glob("*.json")):
        doc = json.loads(p.read_text())
        for ktext, e in (doc.get("entries") or {}).items():
            cls = C.proof_memory_class(e.get("proof"))
            other = 16 if cls == 32 else 32
            if cls in (16, 32) and not (e.get("variants") or {}).get(f"{other}g"):
                picked = (doc["kernel"], ktext, cls, other)
                break
        if picked:
            break
    qual, ktext, cls, other = picked
    ident = f"{qual}::{ktext}"
    print(json.dumps({"on_class": sorted(CC.directory_idents("nvidia/volta", [ident], cls)),
                      "other_class": sorted(CC.directory_idents("nvidia/volta", [ident], other)),
                      "ident": ident}))
''')


def test_an_entry_is_served_on_its_class_only_with_no_card_visible():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO / "src"), "PYTHONNOUSERSITE": "1"}
    r = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["on_class"] == [out["ident"]], out
    assert out["other_class"] == [], out
