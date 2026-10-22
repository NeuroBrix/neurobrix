#!/usr/bin/env python3
"""The shadow BEFORE a copy (inbox 71), v2 (2026-10-03): run under the SERVED venv with the tree's src on the path.
(1) KEYS — a model's rows in the Apple table at the profile's own budget (rungs None or the class's top rung 16 384),
    both Triton modes, each checked with the CERTIFIER'S OWN rule (`entry_covers` for this card's memory class AND
    the running code generator). v1 counted any entry, including the retired generator's, which the runtime does not
    serve (it would sweep at runtime, a miss under certified-only).
(2) MEMORY — the engine's own plan for the cell's request (`--explain-plan --json`, the harness's `cell_request`,
    nothing runs, the container read from the NAS catalogue: kilobytes): on this UNIFIED host a cell holds
    `planned_memory_mb` (the device plan) PLUS `host_footprint.total_bytes`, from one pool; it is compared with the
    memory really available now (`core.host_memory.memory_state`) minus the harness's headroom share (1/16 of the
    host). The harness reserves only the host footprint (told to the Dell 2026-10-03).
Exit 0 = run; 1 = WITHHELD, keys missing (a census defect, certify first); 2 = WITHHELD for memory now (need and
available printed; replayed when the memory is there). usage: precheck_keys_v2.py <tree> <model> <modes csv>"""
import json, os, subprocess, sys
from pathlib import Path
tree, model, modes = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
sys.path.insert(0, str(tree / "src")); sys.path.insert(0, str(tree / "tools"))
from neurobrix.kernels import census_table as T
from neurobrix.kernels import autotune_certified as C
from neurobrix.kernels import autotune_certify as A
from neurobrix.core.host_memory import memory_state

A._bind_hardware_profile()
klass = C.memory_class_gb(A._certifying_device()["memory_mb"])
gen = C.proof_backend({"backend": A._backend()})
vendor, profile = C.active_profile()
rows = T.read(T.table_path(vendor, profile, klass, root=tree / "src" / "neurobrix" / "config" / "census"))
import ast
from neurobrix.triton import autotune_cache as atc
tuners = dict(atc._autotuners())
per = {}
def covered(r):
    # the certifier's own path (`_certify_loop`): the key's output dtype names its file, read once per pass
    qual = r["kernel"]; tuner = tuners[qual]; key = ast.literal_eval(r["key"])
    dtype = C.output_dtype(tuner, key)
    entries = A._file_entries(per.setdefault(qual, {}), dtype, C.file_for(vendor, profile, qual, dtype))
    if C.entry_covers(entries, C.key_repr(key), klass, need_clock=False, need_generator=gen):
        return True
    # the certifier's own rule for a key its class never forms (`_refuse_oversize` -> KeyTooLargeForClass, read
    # by `oversize_for_class`: "no entry is owed, the plan tiles before it"): its tensors exceed the CARD, counted
    # as on this unified device — not owed, so not missing (swin2SR-realworld's 4 416 px conv, 2026-10-04)
    # v3: the certifier's unified rule since mq20 is `_refuse_unified` (the key's phase price against the card);
    # `_refuse_oversize` is the discrete card's and no longer takes `unified=`
    try:
        A._refuse_unified(qual, tuner, key, int(A._certifying_device()["memory_mb"]) * 2**20)
    except A.KeyTooLargeForClass:
        return True
    return False
rc = 0
for mode in modes:
    mine = [r for r in rows if r["model"] == model and r["mode"] == mode
            and (not r.get("rungs_mb") or 16384 in (r.get("rungs_mb") or []))]
    missing = [r for r in mine if not covered(r)]
    print(f"PRECHECK {model} {mode}: {len(mine)} rows at the profile's budget, {len(mine)-len(missing)} served under "
          f"{gen}, {len(missing)} missing" + (f" — first: {missing[0]['kernel'].split('.')[-1]} {missing[0]['key'][:120]}"
                                              if missing else ""), flush=True)
    rc |= bool(missing)
if rc:
    sys.exit(1)
import regression_matrix as R
need_mb = 0
for mode in modes:
    cmd = [sys.executable, "-m", "neurobrix", "run", "--model", model, *R.cell_request(model), *R.MODES[mode],
           "--explain-plan", "--json"]
    env = {**os.environ, "PYTHONPATH": str(tree / "src"), "NEUROBRIX_CACHE": str(Path.home() / "Mounts/Super-NeuroBrix-Cache")}
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    t = p.stdout
    try:
        doc = json.loads(t[t.index("{"):]); plan = doc.get("plan", doc)
    except ValueError:
        tail = ((p.stderr or p.stdout).strip().splitlines() or ["no output"])[-1]
        print(f"PLAN {model} {mode}: no plan JSON (rc {p.returncode}): {tail[:200]} — WITHHELD, a plan refusal is a "
              f"Prism defect, named", flush=True)
        sys.exit(1)
    hf = plan.get("host_footprint") or {}
    dev = float(plan.get("planned_memory_mb") or 0); host = hf.get("total_bytes") or 0
    if "device_bytes" in hf:
        # b859a41f (the Dell, 2026-10-03): on a unified profile the host footprint already carries the device
        # plan (`device_bytes`) — the harness's own reservation is then right, and adding the plan again double counts
        need = host / 2**20
        print(f"PLAN {model} {mode}: {plan.get('strategy')} host footprint {need:.0f} MB (device plan {hf['device_bytes'] / 2**20:.0f} MB inside it)", flush=True)
    else:
        need = dev + host / 2**20
        print(f"PLAN {model} {mode}: {plan.get('strategy')} planned {dev:.0f} MB + host {host / 2**20:.0f} MB = {need:.0f} MB", flush=True)
    need_mb = max(need_mb, need)
st = memory_state()
# the harness's own rule since 67dfafa5 (the Dell, 2026-10-04 00:01): a unified plan (host_footprint.device_bytes)
# is reserved with NO headroom — Prism already left the margin; a discrete plan keeps 1/16 of the host
room = st.available_mb - (0 if "device_bytes" in hf else st.total_mb / 16)
largest = ", ".join(f"{n} {mb} MB" for n, mb in (st.largest_residents or ())[:3])
if need_mb > room:
    print(f"MEMORY {model}: needs {need_mb:.0f} MB, available {st.available_mb} MB - headroom {st.available_mb - room:.0f} = "
          f"{room:.0f} MB ({largest}) — WITHHELD now, replayed when the memory is there", flush=True)
    sys.exit(2)
print(f"MEMORY {model}: needs {need_mb:.0f} MB of {room:.0f} MB available after headroom ({largest}) — run", flush=True)
