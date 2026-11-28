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
    import neurobrix  # the application: nbxtensor's host services
    from nbxtensor.ops import autotune_certified as C
    root = Path("src/neurobrix/config/autotune/nvidia/volta")
    picked = None
    for p in sorted(root.glob("*.json")):
        doc = json.loads(p.read_text())
        for ktext, e in (doc.get("entries") or {}).items():
            cls = C.proof_memory_gb(e.get("proof"))
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


def _child_env() -> dict:
    pp = os.environ.get("PYTHONPATH")
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": f"{REPO / 'src'}{os.pathsep}{pp}" if pp else str(REPO / "src"),
        "PYTHONNOUSERSITE": "1",
    }


def test_an_entry_is_served_on_its_class_only_with_no_card_visible(workshop_configuration):
    r = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO, env=_child_env(), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["on_class"] == [out["ident"]], out
    assert out["other_class"] == [], out


LAZY_CHILD = textwrap.dedent("""
    import json, sys
    from pathlib import Path
    sys.path.insert(0, "tools")
    import certified_census as CC
    import neurobrix  # fresh process: lazily-imported ops (e.g. gemv_vec, addmv_op) are NOT loaded yet
    from nbxtensor.ops import autotune_certified as C
    p = Path("src/neurobrix/config/autotune/apple/apple_m4_pro/addmv_kernel.fp32.json")
    doc = json.loads(p.read_text())
    ktext, entry = next(iter((doc.get("entries") or {}).items()))
    cls = C.proof_memory_gb(entry.get("proof"))
    ident = f"{doc['kernel']}::{ktext}"
    print(json.dumps({"served": sorted(CC.directory_idents("apple/apple_m4_pro", [ident], cls)), "ident": ident}))
""")


def test_a_lazily_imported_kernel_is_counted_as_served_in_a_fresh_census_process(workshop_configuration):
    """Regression: directory_idents used _autotuners() without importing lazily-loaded kernel modules,
    falsely marking already-certified keys (addmv_kernel, gemv_vec_kernel, rmsnorm, layernorm, etc.)
    as to_certify in a fresh census process (47 false positives on apple_m4_pro, 2026-11-27)."""
    r = subprocess.run([sys.executable, "-c", LAZY_CHILD], cwd=REPO, env=_child_env(), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["served"] == [out["ident"]], out


BAD_QUAL_CHILD = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, "tools")
    import certified_census as CC
    served = CC.directory_idents("apple/apple_m4_pro", [".foo::k", "..bar::k", ".a.b::k"], 16)
    print(json.dumps({"served": sorted(served)}))
""")


def test_leading_dot_bad_qualnames_do_not_crash_directory_idents(workshop_configuration):
    """Regression: leading-dot or empty-prefix qualnames must not crash importlib.import_module in directory_idents."""
    r = subprocess.run([sys.executable, "-c", BAD_QUAL_CHILD], cwd=REPO, env=_child_env(), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["served"] == [], out
