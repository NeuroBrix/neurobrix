#!/usr/bin/env python3
"""Retrace, rebuild, gate and re-upload the containers that carry a corrupted
symbolic dimension — resumable, one card, one model at a time.

Owner's word, 2026-09-07: the 98 component graphs (47 containers) carrying an
integer symbolic dim that contradicts the trace are retraced with the corrected
tracer, each in its native dtype, and the zoo re-uploaded — with a gate before
every upload: on its sequential oracle and on its family's locked protocol the
retraced artifact produces a byte-identical output to the old one where the
old was right, and a right output where the old was not, every difference
explained by the closed defect and nothing else. An artifact that fails the
gate is not uploaded; it is a chantier opened and closed first. The old
artifacts stay on the hub (a replace keeps the previous object downloadable);
nothing is deleted. A cut run resumes from its last validated artifact.

Steps per model (state in <out>/<model>/state.json, each idempotent):
  old_outputs  the installed container, run on the sequential oracle and on the
               family protocol (Triton engine) from the given source tree
  trace        forge trace (corrected tracer) into the toolchain's graph cache
  build        forge build → <models-root>/<model>/model.nbx
  backup       the old container copied to <backup>/<model> (restorable)
  install      forge local --overwrite
  new_outputs  the same two runs on the new container
  gate         bytes old vs new on both runs + graph diff (only symbolic_shape
               annotations may differ) → PASS / NEEDS_EXPLANATION / FAIL
  upload       forge replace (an existing hub entry) or forge publish (a new
               one) — only when NEUROBRIX_API_TOKEN is present; else the
               model waits in READY_FOR_UPLOAD

    python tools/retrace_zoo.py --models a,b --gpu 0 --src /path/to/src \
        [--out validation_outputs/retrace_2026_09_07] [--models-root /home/mlops/nbx_builds]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import precision_zoo_campaign as C  # noqa: E402
import repo_env  # noqa: E402  — the repository's .env, loaded the way the build toolchain loads its own

repo_env.load()

PY = "/home/mlops/ml/venv/bin/python"
FORGE = REPO / "forge" / "forge.py"
CACHE = Path.home() / ".neurobrix" / "cache"
HUB_MAP = REPO / "validation_outputs" / "retrace_2026_09_07" / "hub_map.json"
# A container with no hub entry is a NEW publication: its org, name, category, license, tags
# and description are WRITTEN here (the owner's word of 2026-09-07 00:26 for swin2SR-x2), never
# deduced by the tool — a model with neither a hub entry nor a written line is refused by name.
NEW_ENTRIES = REPO / "validation_outputs" / "retrace_2026_09_07" / "new_entries.json"
FAMILIES = REPO / "validation_outputs" / "retrace_2026_09_07" / "families.json"
ANNOTATION_KEYS = {"symbolic_shape"}       # the only tensor fields the closed defect touches
# Trace-time provenance, not the artifact's semantics: the card the trace ran on and its memory
# figures. A retrace on another card must not fail the gate for them (hat-l: 25,632 such fields,
# and the `device` an aten._to_copy's kwargs recorded). The runtime places through Prism.
PROVENANCE_KEYS = {"device", "memory_info", "timestamp_ns"}
# Naming, not semantics: the vendor module an op was recorded under (a vendor rename such as
# `final_layer.norm_final` → `final_layer.final_norm` between two transformers versions moves
# no op and no tensor — VibeVoice's prediction head, 2026-09-07).
NAMING_KEYS = {"parent_module"}
# Derived bookkeeping: the consumers of a tensor are the ops whose inputs name it. The old
# containers carried lists computed before the fusion pass (a norm weight "consumed" by the
# decomposed mul, not the fused rms_norm — Voxtral, VibeVoice, canary 2026-09-07). Never
# compared old-vs-new; the new graph's lists are verified against its own ops instead.
DERIVED_KEYS = {"consumer_op_uids"}


def derived_consumers_consistent(graph: dict) -> int:
    """How many tensors of `graph` carry a consumer list that disagrees with the ops' inputs."""
    ops = graph["ops"] if isinstance(graph.get("ops"), list) else list((graph.get("ops") or {}).values())
    actual = {}
    for op in ops:
        for t in op.get("input_tensor_ids") or []:
            actual.setdefault(t, set()).add(op.get("op_uid"))
    bad = 0
    for tid, m in (graph.get("tensors") or {}).items():
        listed = m.get("consumer_op_uids")
        if listed is None:
            continue
        if set(listed) != actual.get(tid, set()):
            bad += 1
    return bad


def scrub_provenance(node):
    """`node` without its provenance: every `device`/`memory_info` key at any depth, and every
    {"type": "device", ...} leaf (the device argument an op's kwargs recorded)."""
    if isinstance(node, dict):
        if node.get("type") == "device":
            return {"type": "device"}
        return {k: scrub_provenance(v) for k, v in node.items() if k not in PROVENANCE_KEYS and k not in NAMING_KEYS}
    if isinstance(node, list):
        return [scrub_provenance(v) for v in node]
    return node
#: The toolchain's registry key when it differs from the installed container's name (the hub's name).
REGISTRY_ALIAS = {"Sana-1600M-MultiLing": "Sana_1600M_1024px_MultiLing"}
REGISTRY = "https://neurobrix.es"

# The precision policy BOTH arms of the gate run under. A retraced graph carries a new
# signature, so the calibration record embedded in the container (measured on the old graph)
# is refused on it and the runtime takes the conservative path — while the old arm, left to the
# default policy, applies its islands (Kokoro 2026-09-07 05:58: 5.2 dB between the arms with
# identical transcripts; VibeVoice 04:xx: 7.5 dB). One policy on both arms isolates the graph;
# the calibration lever re-measures a retraced container afterwards. Outputs carry the policy
# they were measured under, and the gate compares two arms under the same one, never otherwise.
POLICY_ENV = {"NBX_ACTIVATIONS_FP16_SAFE": "0"}
POLICY = "conservative (NBX_ACTIVATIONS_FP16_SAFE=0)"


def log(msg):
    print(f"[retrace {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else None


def rotate(logfile: Path) -> None:
    """A step's log holds one attempt: an earlier attempt's log is kept aside
    as <name>.<n>.log, so a reader (or a count of the tracer's guard lines)
    never mixes two passes."""
    if not logfile.exists() or logfile.stat().st_size == 0:
        return
    n = 1
    while (logfile.with_name(f"{logfile.stem}.{n}{logfile.suffix}")).exists():
        n += 1
    logfile.rename(logfile.with_name(f"{logfile.stem}.{n}{logfile.suffix}"))


def run(cmd, env, logfile: Path, timeout: int, cwd=None) -> int:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    rotate(logfile)
    with open(logfile, "a") as fh:
        fh.write("$ " + " ".join(str(c) for c in cmd) + "\n"); fh.flush()
        rc = C.run_group(cmd, env, fh, timeout, cwd=cwd)      # its own process group: a timeout kills the children too
        if rc == -9:
            fh.write(f"\n[retrace] TIMEOUT after {timeout} s\n")
        return rc


def _trace_value(v):
    """The trace value a shape argument claims: an integer, or a symbol's trace."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, dict) and "type" in v:                   # a symbol or an expression
        tv = v.get("trace", v.get("trace_value"))
        return tv if isinstance(tv, int) and not isinstance(tv, bool) else None
    return None


_NOT_DIM_TYPES = ("tensor", "tensor_tuple", "dtype", "device", "layout", "list", "int_list", "scalar", "bool", "none")


def _is_dim_node(v) -> bool:
    return isinstance(v, dict) and "type" in v and v.get("type") not in _NOT_DIM_TYPES


def _leaf_diffs(a, b, path=()):
    """Every leaf where two JSON trees of the same structure differ; None when the structure itself differs."""
    if isinstance(a, dict) and isinstance(b, dict):
        if _is_dim_node(a) and _is_dim_node(b):           # two dim expressions: one leaf pair
            return [] if a == b else [(path, a, b)]
        if _is_dim_node(a) and b.get("type") == "scalar" and isinstance(b.get("value"), int):
            return [(path, a, b["value"])]                # a dim expression that became a recorded scalar
        if _is_dim_node(b) and a.get("type") == "scalar" and isinstance(a.get("value"), int):
            return [(path, a["value"], b)]                # a recorded scalar that became a dim expression
        if set(a) != set(b):
            return None
        out = []
        for k in a:
            d = _leaf_diffs(a[k], b[k], path + (k,))
            if d is None:
                return None
            out += d
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return None
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            d = _leaf_diffs(x, y, path + (i,))
            if d is None:
                return None
            out += d
        return out
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        # a symbol or expression {"type": ...} against a bare integer, or two dim expressions
        # spelled differently, are leaf pairs (the classifier judges them)
        if (_is_dim_node(a) and isinstance(b, int) and not isinstance(b, bool)) or \
           (_is_dim_node(b) and isinstance(a, int) and not isinstance(a, bool)):
            return [(path, a, b)]
        if isinstance(a, dict) and a.get("type") == "scalar" and isinstance(a.get("value"), int) and _is_dim_node(b):
            return [(path, a["value"], b)]            # a recorded scalar arg that became a dim expression
        if isinstance(b, dict) and b.get("type") == "scalar" and isinstance(b.get("value"), int) and _is_dim_node(a):
            return [(path, a, b["value"])]            # a dim expression that became a recorded scalar arg
        return None
    return [] if a == b else [(path, a, b)]


def eval_dim(node, env):
    """A dim expression of graph.json evaluated under `env` (symbol id → value); None when the
    grammar is not understood. Symbols, and add/sub/mul/floordiv/mod/neg/max/min over them."""
    if isinstance(node, bool):
        return None
    if isinstance(node, int):
        return node
    if not isinstance(node, dict):
        return None
    t = node.get("type")
    if t == "symbol":
        return env.get(node.get("id"))
    if t == "neg":
        v = eval_dim(node.get("left", node.get("operand")), env)
        return None if v is None else -v
    l = eval_dim(node.get("left"), env)
    r = eval_dim(node.get("right"), env)
    if l is None or r is None:
        return None
    if t == "add":
        return l + r
    if t == "sub":
        return l - r
    if t == "mul":
        return l * r
    if t == "floordiv":
        return None if r == 0 else l // r
    if t == "mod":
        return None if r == 0 else l % r
    if t == "max":
        return max(l, r)
    if t == "min":
        return min(l, r)
    return None


def symbols_of(node, out=None):
    out = {} if out is None else out
    if isinstance(node, dict):
        if node.get("type") == "symbol" and isinstance(node.get("trace"), int):
            out[node["id"]] = node["trace"]
        for v in node.values():
            symbols_of(v, out)
    elif isinstance(node, list):
        for v in node:
            symbols_of(v, out)
    return out


def equivalent_dims(a, b) -> bool:
    """Two dim expressions that agree at the trace assignment and at two others (every symbol
    doubled, then tripled — a multiple of a window stays one): the same extent on the domain
    the container serves, spelled differently by the corrected rules."""
    env0 = {}
    symbols_of(a, env0); symbols_of(b, env0)
    if not env0:
        return False
    for k in (1, 2, 3):
        env = {sid: tv * k for sid, tv in env0.items()}
        va, vb = eval_dim(a, env), eval_dim(b, env)
        if va is None or vb is None or va != vb:
            return False
    return True


def equivalent_modulo_unit_factors(a, b) -> bool:
    """Two dim expressions that agree at every assignment where the unit-trace symbols (a
    batch traced at 1) stay 1 and the others vary — they differ only by factors of those
    symbols: the closed regression's signature (a batch folded, dropped or counted twice)."""
    env0 = {}
    symbols_of(a, env0); symbols_of(b, env0)
    if not env0 or not any(tv == 1 for tv in env0.values()):
        return False
    for k in (1, 2, 3):
        env = {sid: (1 if tv == 1 else tv * k) for sid, tv in env0.items()}
        va, vb = eval_dim(a, env), eval_dim(b, env)
        if va is None or vb is None or va != vb:
            return False
    return True


INT64_MAX = 9223372036854775807


def symbol_remap(old_ctx: dict, new_ctx: dict) -> dict:
    """old symbol id → new symbol id, matched by (name, trace value) in registration
    order — two traces number their symbols independently (canary's perception:
    s0 = batch in June, s0 = seq_len today)."""
    def table(ctx):
        out = {}
        for sid, sym in ((ctx or {}).get("symbols") or {}).items():
            key = (sym.get("name"), sym.get("trace_value", sym.get("trace")))
            out.setdefault(key, []).append(sid)
        return out
    o, n = table(old_ctx), table(new_ctx)
    remap = {}
    for key, olds in o.items():
        news = n.get(key) or []
        for i, sid in enumerate(olds):
            if i < len(news):
                remap[sid] = news[i]
    return remap


def rewrite_symbols(node, remap: dict):
    if isinstance(node, dict):
        if node.get("type") == "symbol" and node.get("id") in remap:
            return {**node, "id": remap[node["id"]]}
        return {k: rewrite_symbols(v, remap) for k, v in node.items()}
    if isinstance(node, list):
        return [rewrite_symbols(v, remap) for v in node]
    return node


def witnessed_arg_changes(old_op: dict, new_op: dict, tensors_new: dict):
    """The differences between two records of one op when each is the closed
    defect at the argument level — two kinds:
    - witnessed: a shape argument (a `size`/`shape` list, or the `args` list it
      mirrors) whose old value — a symbol, or an integer — claimed a trace value
      that contradicts the extent the op's own output tensor witnessed at that
      position, replaced by that witnessed integer;
    - symbolized: an integer argument the old tracer could not express, now the
      symbolic dim the op's own input carries at that extent (the corrected
      rule derived it; a value-matched guess would not be among the inputs'
      dims), with the old integer as its trace value.
    Returns the sites, or None when any difference is of another kind."""
    if {k for k in set(old_op) | set(new_op) if old_op.get(k) != new_op.get(k)} != {"attributes"}:
        return None
    diffs = _leaf_diffs(old_op.get("attributes"), new_op.get("attributes"))
    if not diffs:
        return None
    witnessed = []
    for tid in new_op.get("output_tensor_ids") or []:
        m = tensors_new.get(tid) or {}
        conc = (m.get("symbolic_shape") or {}).get("concrete") or m.get("shape") or []
        if conc:
            witnessed.append(list(conc))
    input_dims = []
    for tid in list(new_op.get("input_tensor_ids") or []) + list(new_op.get("output_tensor_ids") or []):
        m = tensors_new.get(tid) or {}
        for d in (m.get("symbolic_shape") or {}).get("dims") or []:
            if isinstance(d, dict):
                input_dims.append(json.dumps(d, sort_keys=True))
    sites = []
    # BATCH SPLIT RESTORED: the old graph folded a unit-trace symbol (the batch) into its
    # right neighbour — [1, σ·s, …] — and the corrected rule splits it back — [σ, s, …].
    # Recognized as a PAIR of leaves: old[p] == 1 → new[p] = σ (trace 1), and old[p+1] is
    # σ × new[p+1] under every assignment. (2026-08-29 → 09-07 tracer regression.)
    by_parent = {}
    for path, a, b in diffs:
        by_parent.setdefault(path[:-1], {})[path[-1]] = (a, b)
    restored = set()
    for parent_path, slots in by_parent.items():
        for pos, (a, b) in slots.items():
            if a == 1 and isinstance(b, dict) and b.get("type") == "symbol" and _trace_value(b) == 1 and (pos + 1) in slots:
                a2, b2 = slots[pos + 1]
                if isinstance(a2, dict) and _is_dim_node(a2) and equivalent_dims(a2, {"type": "mul", "left": b, "right": b2, "trace": _trace_value(a2)}):
                    restored.add(parent_path + (pos,)); restored.add(parent_path + (pos + 1,))
                    sites.append({"op": new_op.get("op_uid"), "path": ".".join(str(k) for k in parent_path + (pos,)), "old": [a, a2], "new": [b, b2], "kind": "batch-split-restored"})
    for path, a, b in diffs:
        if path in restored:
            continue
        # SLICE END SYMBOLIZED: "to the end" (INT64_MAX) became the extent's own expression —
        # the same slice, now spelled with the dim it ends at (an input's or the output's dim).
        if a == INT64_MAX and isinstance(b, dict) and _is_dim_node(b) and json.dumps(b, sort_keys=True) in input_dims:
            sites.append({"op": new_op.get("op_uid"), "path": ".".join(str(k) for k in path), "old": a, "new": b, "kind": "slice-end-symbolized"})
            continue
        # SLICE END TO THE END: the symmetric spelling — an end that was the dim's own expression
        # is now "to the end" (INT64_MAX); relative by nature, never wrong.
        if b == INT64_MAX and isinstance(a, dict) and _is_dim_node(a) and json.dumps(a, sort_keys=True) in input_dims:
            sites.append({"op": new_op.get("op_uid"), "path": ".".join(str(k) for k in path), "old": a, "new": b, "kind": "slice-end-to-the-end"})
            continue
        # INFERENCE RESTORED: an integer the injection had written into a view became the vendor's
        # -1 again (numel-inferred at runtime; never wrong) — the integer must be the extent.
        if b == -1 and isinstance(a, int) and not isinstance(a, bool) and path and isinstance(path[-1], int):
            _parent = new_op["attributes"]
            for k in path[:-1]:
                _parent = _parent[k]
            if any(len(_parent) == len(c) and c[path[-1]] == a for c in witnessed):
                sites.append({"op": new_op.get("op_uid"), "path": ".".join(str(k) for k in path), "old": a, "new": b, "kind": "inference-restored"})
                continue
        if not path or not isinstance(path[-1], int):
            return None
        pos = path[-1]
        parent = new_op["attributes"]
        for k in path[:-1]:
            parent = parent[k]
        if isinstance(b, dict) and isinstance(a, int) and not isinstance(a, bool):
            # SYMBOLIZED: an integer the old tracer could not express, now the very symbolic
            # dim the op's input or output carries (derived by the rule, never matched by value
            # alone), with the old integer as its trace value and the witnessed extent there.
            # (a vendor's -1, inferred at runtime, may become the derived expression the
            # injection could not build before the rules were fixed: same trace, same extent.)
            tv = _trace_value(b)
            if tv is None or (a != -1 and tv != a) or json.dumps(b, sort_keys=True) not in input_dims:
                return None
            if not any(len(parent) == len(c) and c[pos] == tv for c in witnessed):
                return None
            sites.append({"op": new_op.get("op_uid"), "path": ".".join(str(k) for k in path), "old": a, "new": b, "kind": "symbolized"})
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            ta, tb = _trace_value(a), _trace_value(b)
            # BATCH FACTOR RESTORED: a flatten the old tracer wrote without its batch (E) now
            # carries it (σ·E, σ of trace 1) and is the output's annotated dim — the symmetric
            # case of the split restored (Voxtral's language model, 2026-09-07).
            if (ta is not None and ta == tb and b.get("type") == "mul"
                    and json.dumps(b, sort_keys=True) in input_dims):
                for side, other_side in (("left", "right"), ("right", "left")):
                    sigma = b.get(side)
                    if isinstance(sigma, dict) and sigma.get("type") == "symbol" and _trace_value(sigma) == 1 \
                            and equivalent_dims(b.get(other_side), a):
                        sites.append({"op": new_op.get("op_uid"), "path": ".".join(str(k) for k in path), "old": a, "new": b, "kind": "batch-factor-restored"})
                        break
                else:
                    sigma = None
                if sigma is not None and sites and sites[-1]["kind"] == "batch-factor-restored" and sites[-1]["path"] == ".".join(str(k) for k in path):
                    continue
            # RE-EXPRESSED: the same extent spelled by the corrected rules' algebra — equal at
            # the trace assignment and at two others, and the trace is the witnessed extent.
            if ta is None or ta != tb:
                return None
            if not equivalent_dims(a, b):
                # UNIT FACTOR CORRECTED: equal wherever the batch stays 1, different only by
                # factors of a unit-trace symbol — the old counted the batch twice (whisper-
                # turbo's encoder: s0·(s0·1500) → s0·1500) or not at all; the new is the rule's
                # derivation and must be the op's annotated dim.
                if equivalent_modulo_unit_factors(a, b) and json.dumps(b, sort_keys=True) in input_dims \
                        and any(len(parent) == len(c) and c[pos] == ta for c in witnessed):
                    sites.append({"op": new_op.get("op_uid"), "path": ".".join(str(k) for k in path), "old": a, "new": b, "kind": "unit-factor-corrected"})
                    continue
                return None
            if not any(len(parent) == len(c) and c[pos] == ta for c in witnessed):
                return None
            sites.append({"op": new_op.get("op_uid"), "path": ".".join(str(k) for k in path), "old": a, "new": b, "kind": "re-expressed"})
            continue
        if not isinstance(b, int) or isinstance(b, bool):
            return None
        tv = _trace_value(a)
        if tv is None:
            return None
        if tv == b:
            # UNIT-ONLY EXPRESSION LITERALIZED: an expression whose only symbols are of trace 1
            # is a literal in disguise (parakeet's joint: (batch·50)·(batch·33) for a view of
            # 1650 rows — the batch folded into two frozen lengths); the literal it was worth
            # is admitted at the witnessed extent. A symbol that carries variability never is.
            env0 = symbols_of(a)
            if isinstance(a, dict) and a.get("type") != "symbol" and env0 and all(tv1 == 1 for tv1 in env0.values()) \
                    and any(len(parent) == len(c) and c[pos] == b for c in witnessed):
                sites.append({"op": new_op.get("op_uid"), "path": ".".join(str(k) for k in path), "old": a, "new": b, "kind": "unit-only-literalized"})
                continue
            return None
        if not any(len(parent) == len(c) and c[pos] == b for c in witnessed):
            return None
        sites.append({"op": new_op.get("op_uid"), "path": ".".join(str(k) for k in path), "old": a, "new": b, "kind": "witnessed"})
    return sites


HUB_STORE_HEALTH = "http://10.0.0.36:9000/minio/health/cluster"


def export_readers(cmdlines=None) -> list:
    """The heavy readers of the shared export running on this machine: a trace or a build of
    the build toolchain streams a whole snapshot from the export, and the store — on the same
    storage — then fails its write deadline (a paced 64 MB upload came back SlowDownWrite at
    06:38 on 2026-09-07 while phase B's trace of CogVideoX-2b read its snapshot; the same
    store had taken 192 MB four minutes earlier). Two heavy campaigns never share the export:
    an upload waits for a window with no reader."""
    if cmdlines is None:
        cmdlines = []
        me = os.getpid()
        for d in Path("/proc").iterdir():
            if not d.name.isdigit() or int(d.name) == me:
                continue
            try:
                cmdlines.append((d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace"))
            except OSError:
                continue
    found = []
    for c in cmdlines:
        if "forge.py trace" in c or "forge.py build" in c or "forge.py snap" in c:
            words = c.split()
            what = next((w for w in ("trace", "build", "snap") if f"forge.py {w}" in c), "?")
            model = next((words[i + 1] for i, w in enumerate(words) if w in ("--model", "--name", "--snapshot-path") and i + 1 < len(words)), "?")
            found.append(f"{what} of {Path(model).name}")
    return found


def hub_store_write_probe(org: str, name: str, token: str, registry: str = REGISTRY):
    """200 when the store takes a write today; otherwise the store's own answer, by name.

    The cluster health answered 200 with a write quorum of 1 through every refusal of
    2026-09-07 while each PUT came back 503 `SlowDownWrite` (MinIO's `errErasureWriteQuorum`)
    — after the whole artifact had been streamed (192 MB, then 1.6 GB in parts, every ten
    minutes). Five bytes through the same slot → PUT → drop path say the same thing first."""
    import re
    import requests
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        slot = requests.post(f"{registry}/api/admin/upload", headers=headers, timeout=15,
                             json={"org": org, "name": name, "contentType": "application/octet-stream", "fileSize": 5})
        slot.raise_for_status()
        info = slot.json()
        key, url = info["key"], info["uploadUrl"]
    except Exception as exc:  # noqa: BLE001
        return f"no upload slot from the registry: {type(exc).__name__}: {str(exc)[:120]}"
    try:
        put = requests.put(url, data=b"probe", headers={"Content-Type": "application/octet-stream"}, timeout=60)
        answer = 200 if put.status_code < 400 else None
        if answer is None:
            m = re.search(r"<Code>([^<]+)</Code>(?:.*?<Message>([^<]+)</Message>)?", put.text or "", re.S)
            answer = f"{put.status_code} {m.group(1)}" + (f": {m.group(2)}" if m and m.group(2) else "") if m else f"{put.status_code} on the write probe"
    except Exception as exc:  # noqa: BLE001
        answer = f"write probe: {type(exc).__name__}: {str(exc)[:120]}"
    try:
        requests.delete(f"{registry}/api/admin/upload", headers=headers, params={"key": key}, timeout=15)
    except Exception:  # noqa: BLE001
        pass                                    # the probe object is five bytes under the model's own key prefix
    return answer


def stream_under_probe(url: str, dest: Path, mbps: float, logfile: Path, probe_every: float = 2.0, probe_limit: float = 5.0):
    """`url` streamed into `dest` at no more than `mbps` MB/s; every `probe_every` seconds each
    shared export must list within `probe_limit` seconds, else the stream stops and the export is
    named (None). Returns the bytes written. The URL itself is never written anywhere."""
    import requests
    from snapshot_refresh import _export_answers
    chunk = 1 << 20
    with open(logfile, "a") as fh, requests.get(url, stream=True, timeout=60) as r, open(dest, "wb") as out:
        r.raise_for_status()
        fh.write(f"stream → {dest} at ≤ {mbps} MB/s; export probe every {probe_every} s, limit {probe_limit} s\n"); fh.flush()
        t0 = time.time(); got = 0; last_probe = t0
        for buf in r.iter_content(chunk):
            if not buf:
                continue
            out.write(buf); got += len(buf)
            ahead = got / (mbps * 1e6) - (time.time() - t0)
            if ahead > 0:
                time.sleep(ahead)
            if time.time() - last_probe >= probe_every:
                last_probe = time.time()
                for d in SHARED_STORAGE_EXPORTS:
                    if not os.path.isdir(d) or _export_answers(d, probe_limit) is None:
                        fh.write(f"[probe] {d} did not list within {probe_limit} s after {got / 1e6:.0f} MB — the read stops here, by name\n")
                        return None
        fh.write(f"done: {got} bytes in {time.time() - t0:.0f} s\n")
    return got


# The exports whose storage the hub's object store shares, as mounted on this machine. A path
# the probe cannot see is a probe failure, never a green: `/home/mlops/models` was listed here
# until 06:30 UTC on 2026-09-07 and does not exist — the models export mounts under the repo.
SHARED_STORAGE_EXPORTS = (str(REPO / "models"), str(Path.home() / ".neurobrix" / "cache"), "/home/mlops/hf_snapshots")


def hub_store_health(url: str = HUB_STORE_HEALTH, timeout: float = 10.0, probe_seconds: float = 2.0):
    """200 when the hub object store can take a write; otherwise a reason.

    The store's cluster health answered 200 through every refusal of 2026-09-07:
    it shares its storage with the NFS exports, and it refused or hung exactly
    when they stalled. So the storage behind the store is probed too — an
    export that does not list within `probe_seconds` means the store is under
    the same pressure, and the upload is deferred by name rather than sent
    into a 503 or a hang."""
    import urllib.request, urllib.error, subprocess, time
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            code = r.status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception as e:  # noqa: BLE001
        return type(e).__name__
    if code != 200:
        return code
    for d in SHARED_STORAGE_EXPORTS:
        if not os.path.isdir(d):
            return f"export {d} is not mounted here; the probe cannot see it"
        t = time.time()
        try:
            subprocess.run(["ls", d], capture_output=True, timeout=probe_seconds)
        except subprocess.TimeoutExpired:
            return f"storage under pressure ({d} took more than {probe_seconds:g} s to list)"
        took = time.time() - t
        if took > probe_seconds / 2:
            return f"storage under pressure ({d} answered in {took:.1f} s)"
    return 200


WEIGHT_SUFFIXES = {"safetensors", "bin", "pt", "pth", "gguf", "ckpt", "npz"}


def last_download_event(progress: Path, short: str):
    """The last thing the re-download tool's progress log says about a repository:
    'DONE', 'STOPPED', 'FAILED', 'NOT STARTED', 'downloading', or None when it never
    mentioned it."""
    if not progress.exists():
        return None
    last = None
    for line in progress.read_text(errors="replace").splitlines():
        if f"/{short}: " not in line:
            continue
        rest = line.split(f"/{short}: ", 1)[1]
        for tag in ("DONE", "STOPPED", "FAILED", "NOT STARTED", "downloading", "present"):
            if rest.startswith(tag):
                last = "DONE" if tag == "present" else tag
                break
    return last


def snapshot_has_a_format(p: Path) -> bool:
    """The layouts the toolchain's format detector accepts: a diffusers pipeline
    (model_index.json), a transformers model (config.json), a NeMo archive (*.nemo)
    or a NeMo directory (model_config.yaml + model_weights.ckpt)."""
    return ((p / "model_index.json").exists() or (p / "config.json").exists()
            or any(p.glob("*.nemo"))
            or ((p / "model_config.yaml").exists() and (p / "model_weights.ckpt").exists()))


def snapshot_weight_gb(snap) -> float:
    """The weight files of a snapshot, in GB (what a build stages)."""
    total = 0
    for f in Path(snap).rglob("*"):
        if f.is_file() and f.suffix.lstrip(".") in WEIGHT_SUFFIXES:
            total += f.stat().st_size
    return total / 1e9


class Model:
    def __init__(self, name: str, args):
        self.name = name
        self.args = args
        self.dir = Path(args.out) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"model": name, "steps": {}}
        fams = json.loads(FAMILIES.read_text()) if FAMILIES.exists() else {}
        self.family = fams.get(name) or C.family_of(name)
        hub = json.loads(HUB_MAP.read_text()) if HUB_MAP.exists() else {}
        self.hub = hub.get(name)                       # "org/name" or None (a new entry)
        self.registry_name = REGISTRY_ALIAS.get(name, name)
        self.new_name = (self.state["steps"].get("install") or {}).get("installed_name") or name

    def done(self, step):
        st = self.state["steps"].get(step) or {}
        if st.get("ok") is not True:
            return False
        if step in ("old_outputs", "new_outputs") and st.get("policy") != POLICY:
            return False                       # measured under another policy: another arm, re-run
        return True
    def mark(self, step, ok, **info):
        self.state["steps"][step] = {"ok": ok, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}
        self.state_path.write_text(json.dumps(self.state, indent=1))

    # -- environment -------------------------------------------------------
    def env(self, tree: bool = True):
        e = {**os.environ}
        if self.args.gpu not in (None, ""):
            e["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
        if tree and self.args.src:
            e["PYTHONPATH"] = str(Path(self.args.src).resolve())
        e["TMPDIR"] = str(Path(self.args.tmp).resolve()); Path(self.args.tmp).mkdir(parents=True, exist_ok=True)
        e["NEUROBRIX_MODELS_ROOT"] = str(Path(self.args.models_root).resolve()); Path(self.args.models_root).mkdir(parents=True, exist_ok=True)
        return e

    # -- the two runs of the locked protocol --------------------------------
    def outputs(self, tag: str) -> dict:
        """The sequential oracle and the family protocol on the Triton engine, outputs hashed."""
        name = self.name if tag == "old" else self.new_name
        req = C.request_args(self.name, self.family, list(self.args.extra))
        ext = C.output_ext(self.family, req)
        res = {}
        for arm, flag in (("sequential", ["--sequential"]), ("triton", ["--triton"])):
            outp = self.dir / f"{tag}_{arm}{ext}"
            if outp.exists() and (self.dir / f"{tag}_{arm}.log").exists() and sha(outp):
                res[arm] = {"rc": 0, "sha": sha(outp), "output": str(outp), "cached": True}
                continue
            cmd = [PY, "-c", "import sys; from neurobrix.cli import main; sys.exit(main())", "run", "--model", name] + req + flag + ["--output", str(outp)]
            t0 = time.time()
            # One precision policy on both arms (POLICY above).
            env = self.env(); env.update(POLICY_ENV)
            rc = run(cmd, env, self.dir / f"{tag}_{arm}.log", self.args.timeout)
            logtext = (self.dir / f"{tag}_{arm}.log").read_text(errors="replace")
            unsupported = "UNSUPPORTED PATH" in logtext and "encoding" in logtext
            res[arm] = {"rc": rc, "sha": sha(outp), "output": str(outp), "seconds": round(time.time() - t0, 1),
                        "n_a": bool(unsupported and arm == "sequential")}
        return res

    # -- steps --------------------------------------------------------------
    def set_aside(self, prefix: str, why: str) -> None:
        """Outputs of another attempt or another policy are kept aside, never mixed with this one's."""
        stale = [f for f in sorted(self.dir.glob(f"{prefix}*")) if f.is_file()]
        if not stale:
            return
        keep = self.dir / f"superseded_{time.strftime('%H%M%S')}_{prefix.rstrip('_')}"
        keep.mkdir(exist_ok=True)
        for f in stale:
            f.rename(keep / f.name)
        (keep / "WHY.txt").write_text(why + "\n")

    def cache_holds_backup(self):
        """True when the cache's container is the one the backup holds (same build), False when
        another build sits there, None without a backup or a cache. The state cannot answer this:
        a state reset for a re-trace drops its install step while the cache keeps the build."""
        cm = CACHE / self.name / "manifest.json"
        bm = Path(self.args.backup) / self.name / "manifest.json"
        if not (cm.exists() and bm.exists()):
            return None
        return json.loads(cm.read_text()).get("created_at") == json.loads(bm.read_text()).get("created_at")

    def step_old_outputs(self):
        if self.done("old_outputs"): return True
        st = self.state["steps"].get("old_outputs") or {}
        if st.get("ok") is True:
            self.set_aside("old_", f"measured under the policy '{st.get('policy') or 'default'}'; "
                                   f"the gate runs both arms under '{POLICY}'")
        restored = False
        if self.cache_holds_backup() is False:
            # The cache holds another build (the retraced container, or a pass's): the old arm
            # runs on the hub's previous object, brought back through the standard install.
            if not self.restore_previous():
                return False
            restored = True
        elif not (CACHE / self.name / "manifest.json").exists():
            self.mark("old_outputs", False, error="no installed container"); return False
        res = self.outputs("old")
        ok = all(v["rc"] == 0 or v.get("n_a") for v in res.values())
        self.mark("old_outputs", ok, runs=res, policy=POLICY,
                  container="the hub's previous object" if restored else "the installed container")
        if restored and self.done("build") and self.done("install"):
            ok = self.reinstall_new() and ok    # the retraced container goes back into the cache
        return ok

    def restore_previous(self) -> bool:
        """The hub's CURRENT object — the container before this retrace, nothing was replaced yet —
        installed back into the cache for the old arm, through the standard install. Read through
        the admin URL (the public endpoint counts downloads), streamed under the rate cap with the
        export probe: the store shares its storage with the exports, and a stream beside the
        batteries' reads stalled them six times on 2026-09-07. The object's identity is checked
        against the backup (same build) before it is installed."""
        if not self.hub:
            self.mark("old_outputs", False, error="no hub entry to restore the previous object from"); return False
        try:
            repo_env.require("NEUROBRIX_API_TOKEN")
        except repo_env.MissingVariable as exc:
            self.mark("old_outputs", False, state="REFUSED", reason=str(exc))
            log(f"{self.name}: restore {exc}"); return False
        health = hub_store_health()
        if health != 200:
            self.mark("old_outputs", False, state="DEFERRED", reason=f"hub object store: {health}; the previous object is read when it answers 200")
            log(f"{self.name}: restore DEFERRED — hub object store: {health}"); return False
        import zipfile
        import requests                                # the client the build toolchain publishes with (the edge refuses urllib's agent)
        org, name = self.hub.split("/", 1)
        logfile = self.dir / "restore.log"
        rotate(logfile)
        t0 = time.time()
        try:
            r = requests.get(f"{REGISTRY}/api/models/{org}/{name}", timeout=30)
            r.raise_for_status()
            body = r.json(); rec = body.get("model", body)
            key, size = rec.get("fileUrl"), int(rec.get("fileSize") or 0)
            with open(logfile, "a") as fh:
                fh.write(f"previous object of {self.hub}: {key} ({size} bytes, updated {rec.get('updatedAt')})\n")
            v = requests.get(f"{REGISTRY}/api/admin/upload", params={"key": key}, timeout=30,
                             headers={"Authorization": f"Bearer {os.environ['NEUROBRIX_API_TOKEN']}"})
            v.raise_for_status()
            url = v.json()["url"]                       # a read URL: never written anywhere
        except Exception as exc:  # noqa: BLE001 — named, never a traceback that ends the chain
            reason = f"the hub did not give the previous object's read URL: {type(exc).__name__}: {str(exc)[:160]}"
            self.mark("old_outputs", False, state="DEFERRED", reason=reason)
            log(f"{self.name}: restore DEFERRED — {reason}"); return False
        dest = Path(self.args.tmp) / "previous" / self.name / "model.nbx"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            got = stream_under_probe(url, dest, self.args.restore_mbps, logfile)
        except Exception as exc:  # noqa: BLE001
            reason = f"the read of the previous object failed: {type(exc).__name__}: {str(exc)[:160]}"
            self.mark("old_outputs", False, state="DEFERRED", reason=reason)
            log(f"{self.name}: restore DEFERRED — {reason}"); return False
        if got is None:
            self.mark("old_outputs", False, state="DEFERRED", reason="an export stopped answering during the read of the previous object (restore.log names it)")
            log(f"{self.name}: restore DEFERRED — an export stopped answering during the read; stopped by name"); return False
        if got != size:
            self.mark("old_outputs", False, error=f"previous object: {got} bytes read, the hub records {size}")
            log(f"{self.name}: restore FAILED — {got} bytes read, the hub records {size}"); return False
        with zipfile.ZipFile(dest) as zf:
            created = json.loads(zf.read("manifest.json")).get("created_at")
        backed = json.loads((Path(self.args.backup) / self.name / "manifest.json").read_text()).get("created_at")
        if created != backed:
            self.mark("old_outputs", False, error=f"the hub's object is not the container that was backed up (built {created}, backup {backed})")
            log(f"{self.name}: restore FAILED — the hub's object was built {created}, the backup {backed}"); return False
        rc = run([PY, str(FORGE), "local", str(dest), "--overwrite"], self.env(tree=False), self.dir / "restore_install.log", 3600, cwd=str(REPO / "forge"))
        ok = rc == 0 and self.cache_holds_backup() is True
        self.state["steps"]["previous_object"] = {"ok": ok, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "key": key, "bytes": got,
                                                  "built": created, "seconds": round(time.time() - t0, 1), "rc": rc}
        self.state_path.write_text(json.dumps(self.state, indent=1))
        if not ok:
            self.mark("old_outputs", False, error=f"the previous object did not install as the backed-up container (rc {rc})")
            log(f"{self.name}: restore FAILED — the previous object did not install as the backed-up container (rc {rc})")
            return False
        log(f"{self.name}: previous object restored from the hub ({got / 1e6:.0f} MB in {time.time() - t0:.0f} s, built {created}); the old arm runs on it")
        return True

    def reinstall_new(self) -> bool:
        """The retraced container back into the cache after the old arm ran on the previous object."""
        nbx = (self.state["steps"].get("build") or {}).get("nbx")
        if not nbx or not Path(nbx).exists():
            self.mark("install", False, error="the retraced .nbx is no longer staged; the chain rebuilds it"); return False
        rc = run([PY, str(FORGE), "local", nbx, "--overwrite"], self.env(tree=False), self.dir / "install.log", 3600, cwd=str(REPO / "forge"))
        ok = rc == 0 and (CACHE / self.new_name / "manifest.json").exists() and self.cache_holds_backup() is False
        self.mark("install", ok, rc=rc, installed_name=self.new_name, reinstalled="after the old arm ran on the hub's previous object")
        shutil.rmtree(Path(self.args.tmp) / "previous" / self.name, ignore_errors=True)
        return ok

    def snapshot(self):
        """The model's COMPLETE snapshot: the export first, then the toolchain's own
        download directory. A directory with files in it is what a stopped download
        leaves behind (Sana 4K: 6 GB, no model_index.json, 2026-09-07): complete =
        the format's index file present, no partial file, and — for a repository the
        re-download tool touched — the toolchain's completion marker."""
        snap_logs = Path(self.args.out) / "snap"
        for root in (Path("/home/mlops/hf_snapshots"), Path.home() / ".cache" / "neurobrix" / "hf_snapshots"):
            for nm in (self.registry_name, self.name):
                p = root / nm
                if not (p.is_dir() and any(p.iterdir())):
                    continue
                if not snapshot_has_a_format(p):
                    continue
                if any(p.rglob("*.incomplete")):
                    continue
                # A repository whose LAST event in the re-download tool's progress log is not
                # DONE (downloading, stopped by the probe, failed, not started) is incomplete
                # unless the toolchain's marker says otherwise; one with no event there is
                # judged by its files alone (the chains before the marker existed).
                if not (p / ".snapshot_complete").exists() and last_download_event(snap_logs / "progress.log", nm) not in (None, "DONE"):
                    continue
                return p
        return None

    def step_trace(self):
        if self.done("trace"): return True
        snap = self.snapshot()
        if snap is None:
            self.mark("trace", False, error="no COMPLETE snapshot on the export or in the download directory (index file, no partial file, the marker when the tool touched it)"); return False
        t0 = time.time()
        cmd = [PY, str(FORGE), "trace", "--model", self.registry_name, "--family", self.family, "--device", "cuda:0", "--path", str(snap)]
        rc = run(cmd, self.env(tree=False), self.dir / "trace.log", self.args.trace_timeout, cwd=str(REPO / "forge"))
        self.mark("trace", rc == 0, rc=rc, seconds=round(time.time() - t0, 1))
        return rc == 0

    def step_build(self):
        if self.done("build"): return True
        snap = self.snapshot()
        if snap is None:
            self.mark("build", False, error="no snapshot on the export or in the download directory"); return False
        # The staging disk must hold the container twice (the build's staging and the .nbx) with
        # headroom; a build that would fill it is deferred by name (the root fs filled at 04:21).
        need_gb = 2 * snapshot_weight_gb(snap) + 10
        free_gb = shutil.disk_usage(self.args.models_root).free / 1e9
        if free_gb < need_gb:
            self.mark("build", False, state="DEFERRED", reason=f"staging disk: {free_gb:.0f} GB free, {need_gb:.0f} GB needed (twice the snapshot's weights + 10)")
            log(f"{self.name}: build DEFERRED — staging disk {free_gb:.0f} GB free, {need_gb:.0f} GB needed; uploads must drain first")
            return False
        t0 = time.time()
        cmd = [PY, str(FORGE), "build", "--snapshot-path", str(snap), "--family", self.family, "--overwrite"]
        rc = run(cmd, self.env(tree=False), self.dir / "build.log", self.args.trace_timeout, cwd=str(REPO / "forge"))
        root = Path(self.args.models_root)
        found = None                                   # the builder writes <models-root>/<family>/<name>/model.nbx
        for nm in (self.registry_name, self.name):
            for cand in [root / self.family / nm / "model.nbx", root / nm / "model.nbx"] + sorted(root.glob(f"*/{nm}/*.nbx")):
                if cand.exists():
                    found = cand; break
            if found:
                break
        ok = rc == 0 and found is not None
        self.mark("build", ok, rc=rc, nbx=str(found) if found else None, seconds=round(time.time() - t0, 1),
                  gb=round(found.stat().st_size / 2**30, 2) if found else None)
        return ok

    def step_backup(self):
        if self.done("backup"): return True
        src = CACHE / self.name
        dst = Path(self.args.backup) / self.name
        if dst.exists() and (dst / "manifest.json").exists():
            self.mark("backup", True, path=str(dst), cached=True); return True
        t0 = time.time()
        # The backup keeps what the gate reads — manifest, topology, defaults, every component's
        # graph, profile and index — and not the weights: those stay on the hub as the previous
        # object (the rollback `forge replace` keeps). Fourteen full copies filled 31 GB of the
        # staging disk on 2026-09-07 for 401 MB of graphs.
        def _no_weights(_dir, names):
            return [n for n in names if n.rsplit(".", 1)[-1] in WEIGHT_SUFFIXES]
        shutil.copytree(src, dst, symlinks=True, ignore=_no_weights)
        self.mark("backup", True, path=str(dst), seconds=round(time.time() - t0, 1), weights="on the hub (previous object)")
        return True

    def step_install(self):
        if self.done("install"): return True
        nbx = (self.state["steps"].get("build") or {}).get("nbx")
        if not nbx or not Path(nbx).exists():
            self.mark("install", False, error="no built .nbx"); return False
        import zipfile
        try:
            with zipfile.ZipFile(nbx) as zf:
                installed = json.loads(zf.read("manifest.json")).get("model_name") or self.name
        except Exception:
            installed = self.name
        rc = run([PY, str(FORGE), "local", nbx, "--overwrite"], self.env(tree=False), self.dir / "install.log", 3600, cwd=str(REPO / "forge"))
        ok = rc == 0 and (CACHE / installed / "manifest.json").exists()
        self.new_name = installed
        self.mark("install", ok, rc=rc, installed_name=installed)
        return ok

    def step_new_outputs(self):
        if self.done("new_outputs"): return True
        # A previous attempt's outputs describe a previous container (or another policy): never reused as this one's.
        st = self.state["steps"].get("new_outputs") or {}
        self.set_aside("new_", "a previous attempt's outputs" if st.get("policy") == POLICY else
                       f"measured under the policy '{st.get('policy') or 'default'}'; the gate runs both arms under '{POLICY}'")
        res = self.outputs("new")
        ok = all(v["rc"] == 0 or v.get("n_a") for v in res.values())
        self.mark("new_outputs", ok, runs=res, policy=POLICY)
        return ok

    def graph_diff(self) -> dict:
        """Old vs new graph.json per component: every difference must be the
        closed defect — the symbolic-shape annotation, or a shape argument whose
        false symbol the corrected tracer replaced by the extent its own output
        witnessed (`witnessed_arg_changes`); anything else — an op, a shape, a
        dtype, another attribute — is a difference the gate refuses."""
        old_root = Path(self.args.backup) / self.name / "components"
        new_root = CACHE / self.new_name / "components"
        report = {"components": {}, "beyond_annotation": 0, "annotation_changes": 0, "arg_witnessed": 0, "pruned_dead_ops": 0, "corrupted_before": 0, "corrupted_after": 0}
        for comp_dir in sorted(new_root.glob("*")):
            og, ng = old_root / comp_dir.name / "graph.json", comp_dir / "graph.json"
            if not og.exists() or not ng.exists():
                report["components"][comp_dir.name] = {"error": "graph missing on one side"}; report["beyond_annotation"] += 1; continue
            o, n = json.loads(og.read_text()), json.loads(ng.read_text())
            ops_o = o["ops"] if isinstance(o.get("ops"), list) else list((o.get("ops") or {}).values())
            ops_n = n["ops"] if isinstance(n.get("ops"), list) else list((n.get("ops") or {}).values())
            rec = {"ops_old": len(ops_o), "ops_new": len(ops_n), "op_diffs": 0, "tensor_diffs_beyond": 0, "annotation_changes": 0,
                   "arg_witnessed": 0, "arg_kinds": {}, "arg_witnessed_sites": [], "pruned_dead_ops": 0, "corrupted_before": 0, "corrupted_after": 0}
            to, tn = o.get("tensors") or {}, n.get("tensors") or {}
            # Ops align by uid. An op only the old graph carries, whose outputs no old op consumed
            # and no graph output named, is a DEAD op the corrected tracer prunes (R19: the DAG
            # holds compute only — canary's `arange` and five `empty`, 2026-09-07): admitted and
            # counted. An op only the new graph carries, or an old op with a consumer, is beyond.
            remap = symbol_remap(o.get("symbolic_context"), n.get("symbolic_context"))
            if remap and any(k != v for k, v in remap.items()):
                ops_o = [rewrite_symbols(x, remap) for x in ops_o]
                to = {tid: rewrite_symbols(m, remap) for tid, m in to.items()}
                rec["symbols_remapped"] = sum(1 for k, v in remap.items() if k != v)
            by_o = {x.get("op_uid"): x for x in ops_o}
            by_n = {x.get("op_uid"): x for x in ops_n}
            consumed_o = {t for x in ops_o for t in (x.get("input_tensor_ids") or [])}
            outputs_o = set(o.get("outputs") or o.get("output_tensor_ids") or [])
            for uid, a in by_o.items():
                if uid in by_n:
                    continue
                outs = a.get("output_tensor_ids") or []
                if outs and not any(t in consumed_o or t in outputs_o for t in outs):
                    rec["pruned_dead_ops"] += 1
                else:
                    rec["op_diffs"] += 1
            rec["op_diffs"] += sum(1 for uid in by_n if uid not in by_o)
            for uid, a in by_o.items():
                b = by_n.get(uid)
                if b is None:
                    continue
                a = scrub_provenance(a)
                b = scrub_provenance(b)
                if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                    sites = witnessed_arg_changes(a, b, tn)
                    if sites is None:
                        rec["op_diffs"] += 1
                    else:
                        rec["arg_witnessed"] += len(sites)
                        for x in sites:
                            rec["arg_kinds"][x["kind"]] = rec["arg_kinds"].get(x["kind"], 0) + 1
                        rec["arg_witnessed_sites"] = (rec["arg_witnessed_sites"] + sites)[:20]
            dead_out = {t for uid, x in by_o.items() if uid not in by_n for t in (x.get("output_tensor_ids") or [])}
            for tid in set(to) | set(tn):
                a, b = to.get(tid), tn.get(tid)
                if a is None or b is None:
                    if tid in dead_out and b is None:
                        continue                       # the pruned dead op's own output
                    rec["tensor_diffs_beyond"] += 1; continue
                for k in set(a) | set(b):
                    if k in PROVENANCE_KEYS or k in DERIVED_KEYS:
                        continue
                    if a.get(k) != b.get(k):
                        if k in ANNOTATION_KEYS:
                            rec["annotation_changes"] += 1
                        else:
                            rec["tensor_diffs_beyond"] += 1
            for tens, key in ((to, "corrupted_before"), (tn, "corrupted_after")):
                for m in tens.values():
                    dims = (m.get("symbolic_shape") or {}).get("dims") or []; shp = m.get("shape") or []
                    if any(isinstance(d, int) and not isinstance(d, bool) and i < len(shp) and isinstance(shp[i], int) and d != shp[i] for i, d in enumerate(dims)):
                        rec[key] += 1
            rec["derived_inconsistent"] = derived_consumers_consistent(n)
            rec["tensor_diffs_beyond"] += rec["derived_inconsistent"]
            report["components"][comp_dir.name] = rec
            report["beyond_annotation"] += rec["op_diffs"] + rec["tensor_diffs_beyond"]
            report["annotation_changes"] += rec["annotation_changes"]
            report["arg_witnessed"] += rec["arg_witnessed"]
            report["pruned_dead_ops"] += rec["pruned_dead_ops"]
            report["corrupted_before"] += rec["corrupted_before"]; report["corrupted_after"] += rec["corrupted_after"]
        return report

    def step_gate(self):
        if self.done("gate"): return True
        so = self.state["steps"].get("old_outputs") or {}
        sn = self.state["steps"].get("new_outputs") or {}
        if so.get("policy") != sn.get("policy"):
            reason = (f"the two arms ran under different precision policies: old '{so.get('policy') or 'default'}', "
                      f"new '{sn.get('policy') or 'default'}' — a difference between them would not be the graph's")
            self.mark("gate", False, verdict="FAIL", reason=reason)
            log(f"{self.name}: gate FAIL — {reason}"); return False
        old, new = so.get("runs") or {}, sn.get("runs") or {}
        bytes_verdict = {}
        for arm in ("sequential", "triton"):
            o, nn = old.get(arm) or {}, new.get(arm) or {}
            if o.get("n_a") and nn.get("n_a"):
                bytes_verdict[arm] = "N/A (no ATen oracle for an encoded build)"; continue
            if o.get("rc") != 0 or nn.get("rc") != 0:
                bytes_verdict[arm] = f"FAILED (old rc {o.get('rc')}, new rc {nn.get('rc')})"; continue
            same = o.get("sha") and o.get("sha") == nn.get("sha")
            if same:
                bytes_verdict[arm] = "IDENTICAL"
            else:
                try:
                    d = C.gate(Path(o["output"]), Path(nn["output"]))
                except Exception as e:  # noqa: BLE001
                    d = {"error": str(e)}
                bytes_verdict[arm] = {"DIFFERENT": d}
        gd = self.graph_diff()
        identical = all(v == "IDENTICAL" or str(v).startswith("N/A") for v in bytes_verdict.values())
        failed = any(str(v).startswith("FAILED") for v in bytes_verdict.values())
        if failed or gd["beyond_annotation"] or gd["corrupted_after"]:
            verdict = "FAIL"
        elif identical and gd["corrupted_before"] > 0:
            verdict = "PASS"                     # the old was right; the new is byte-identical and carries no corrupted dim
        elif identical:
            verdict = "PASS (no corrupted dim was present in the old container)"
        else:
            verdict = "NEEDS_EXPLANATION"        # bytes differ: the difference must be the closed defect and nothing else
        self.mark("gate", verdict.startswith("PASS"), verdict=verdict, bytes=bytes_verdict, graph=gd, policy=POLICY)
        log(f"{self.name}: gate {verdict} — both arms under {POLICY}; bytes {bytes_verdict}; graph: {gd['annotation_changes']} annotation change(s), "
            f"{gd['arg_witnessed']} shape argument(s) of the closed defect "
            f"(witnessed {sum(r.get('arg_kinds', {}).get('witnessed', 0) for r in gd['components'].values())}, "
            f"symbolized {sum(r.get('arg_kinds', {}).get('symbolized', 0) for r in gd['components'].values())}, "
            f"re-expressed {sum(r.get('arg_kinds', {}).get('re-expressed', 0) for r in gd['components'].values())}, "
            f"batch split restored {sum(r.get('arg_kinds', {}).get('batch-split-restored', 0) for r in gd['components'].values())}, "
            f"batch factor restored {sum(r.get('arg_kinds', {}).get('batch-factor-restored', 0) for r in gd['components'].values())}, "
            f"unit factor corrected {sum(r.get('arg_kinds', {}).get('unit-factor-corrected', 0) for r in gd['components'].values())}, "
            f"slice end symbolized {sum(r.get('arg_kinds', {}).get('slice-end-symbolized', 0) for r in gd['components'].values())}, "
            f"slice end to the end {sum(r.get('arg_kinds', {}).get('slice-end-to-the-end', 0) for r in gd['components'].values())}, "
            f"inference restored {sum(r.get('arg_kinds', {}).get('inference-restored', 0) for r in gd['components'].values())}, "
            f"unit-only literalized {sum(r.get('arg_kinds', {}).get('unit-only-literalized', 0) for r in gd['components'].values())}), "
            f"{gd['pruned_dead_ops']} dead op(s) pruned, "
            f"{gd['beyond_annotation']} beyond, corrupted dims {gd['corrupted_before']} → {gd['corrupted_after']}")
        return verdict.startswith("PASS")

    def step_upload(self):
        if self.done("upload"): return True
        nbx = (self.state["steps"].get("build") or {}).get("nbx")
        try:
            repo_env.require("NEUROBRIX_API_TOKEN")
        except repo_env.MissingVariable as exc:
            # An explicit refusal that names the variable and the file — never a state that waits without saying why.
            self.mark("upload", False, state="REFUSED", reason=str(exc), nbx=nbx)
            log(f"{self.name}: upload {exc}")
            return False
        readers = export_readers()
        if readers:
            self.mark("upload", False, state="DEFERRED", reason=f"the export is read by {', '.join(readers)}; the upload waits for a window with no reader", nbx=nbx)
            log(f"{self.name}: upload DEFERRED — the export is read by {', '.join(readers)}; the upload waits for a window with no reader")
            return False
        health = hub_store_health()
        if health != 200:
            # The hub's object store refuses writes (its cluster health answers other than 200):
            # said by name, the artifact stays ready, the chain moves on; a later pass uploads it.
            self.mark("upload", False, state="DEFERRED", reason=f"hub object store: {health}; retry when it answers 200", nbx=nbx)
            log(f"{self.name}: upload DEFERRED — hub object store: {health}; the artifact is gated and ready, a later pass uploads it")
            return False
        if self.hub:
            org, name = self.hub.split("/", 1)
            probe = hub_store_write_probe(org, name, os.environ["NEUROBRIX_API_TOKEN"])
            if probe != 200:
                # The store refuses writes: said with the store's own name, before any artifact is streamed.
                self.mark("upload", False, state="DEFERRED", reason=f"hub object store refuses writes: {probe}; retry when a five-byte write lands", nbx=nbx)
                log(f"{self.name}: upload DEFERRED — the store refuses writes: {probe}; the artifact is gated and ready, a later pass uploads it")
                return False
            cmd = [PY, str(FORGE), "replace", "--org", org, "--name", name, nbx]
        else:
            new = (json.loads(NEW_ENTRIES.read_text()) if NEW_ENTRIES.exists() else {}).get(self.name)
            if not new:
                self.mark("upload", False, state="REFUSED", reason=f"no hub entry and no written new-entry line for {self.name} in {NEW_ENTRIES.name}", nbx=nbx)
                log(f"{self.name}: upload REFUSED — no hub entry and no written new-entry line in {NEW_ENTRIES.name}")
                return False
            cmd = [PY, str(FORGE), "publish", nbx, "--org", new["org"], "--name", new["name"], "--category", new["category"],
                   "--description", new["description"], "--tags", new["tags"], "--license", new["license"]]
        if self.args.upload_mbps > 0:
            # The store writes each block as it arrives and answered a 548 MB/s burst with
            # SlowDownWrite after the whole artifact had streamed (2026-09-07): paced.
            cmd += ["--max-write-mbps", str(self.args.upload_mbps)]
        rc = run(cmd, self.env(tree=False), self.dir / "upload.log", 7200, cwd=str(REPO / "forge"))
        if rc == 0 and not self.hub and NEW_ENTRIES.exists():
            new = json.loads(NEW_ENTRIES.read_text()).get(self.name) or {}
            if new:
                hub = json.loads(HUB_MAP.read_text()) if HUB_MAP.exists() else {}
                hub[self.name] = self.hub = f"{new['org']}/{new['name']}"
                HUB_MAP.write_text(json.dumps(hub, indent=1, sort_keys=True))
        self.mark("upload", rc == 0, rc=rc, command=" ".join(cmd[2:]))
        if rc == 0 and nbx and Path(nbx).exists():
            # The hub holds it now (checksum verified by the toolchain before the repoint);
            # the staged copy on the root fs is the space the next build needs (the root fs
            # filled up on 2026-09-07 with seven staged containers).
            import shutil
            shutil.rmtree(Path(nbx).parent, ignore_errors=True)
            log(f"{self.name}: staged build removed after the upload ({nbx})")
        return rc == 0

    def run_all(self):
        order = [("old_outputs", self.step_old_outputs), ("trace", self.step_trace), ("build", self.step_build),
                 ("backup", self.step_backup), ("install", self.step_install), ("new_outputs", self.step_new_outputs),
                 ("gate", self.step_gate), ("upload", self.step_upload)]
        for name, fn in order:
            if self.done(name):
                continue
            log(f"{self.name}: {name} …")
            ok = fn()
            if not ok:
                st = self.state["steps"].get(name) or {}
                log(f"{self.name}: {name} → {st.get('state') or 'STOPPED'} ({st.get('verdict') or st.get('error') or st.get('reason') or ('rc ' + str(st.get('rc')))})")
                return False
        log(f"{self.name}: complete")
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True)
    ap.add_argument("--gpu", default=None, help="the card(s) to pin: an ordinal or a comma list (a two-card trace)")
    ap.add_argument("--src", default=None, help="the engine source tree the runs use (PYTHONPATH)")
    ap.add_argument("--out", default=str(REPO / "validation_outputs" / "retrace_2026_09_07"))
    ap.add_argument("--models-root", default="/home/mlops/nbx_builds", help="where forge build writes the .nbx (root fs, not the export)")
    ap.add_argument("--tmp", default="/home/mlops/nbx_tmp", help="TMPDIR for the build staging (root fs)")
    ap.add_argument("--backup", default=str(Path.home() / ".neurobrix" / "backups" / "retrace_2026_09_07"))
    ap.add_argument("--extra", default="", help="extra request args for the family protocol")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--trace-timeout", type=int, default=14400)
    ap.add_argument("--upload-mbps", type=float, default=10.0,
                    help="pace an upload to this many MB/s (0 = unpaced); the store refuses a burst its drive cannot absorb")
    ap.add_argument("--restore-mbps", type=float, default=10.0,
                    help="the rate cap on a read of the hub's previous object (its store shares the exports' storage)")
    ap.add_argument("--stop-at", default=None, help="stop after this step (e.g. gate)")
    ap.add_argument("--only-upload", action="store_true",
                    help="run the upload step only, for a container whose gate is PASS; anything else is refused by name "
                         "(an upload loop must never trace or build — a reset state once made one trace Kokoro beside a pass)")
    args = ap.parse_args()
    import shlex
    args.extra = shlex.split(args.extra)
    summary = {}
    for m in [x for x in args.models.split(",") if x]:
        model = Model(m, args)
        if args.only_upload:
            if (model.state["steps"].get("gate") or {}).get("verdict", "").startswith("PASS") and model.done("gate"):
                if not model.done("upload"):
                    log(f"{m}: upload …")
                    model.step_upload()
            else:
                log(f"{m}: upload REFUSED — the gate is not PASS in the state (an upload loop never traces or builds)")
            summary[m] = {k: (v.get("verdict") or v.get("state") or ("ok" if v.get("ok") else "failed")) for k, v in model.state["steps"].items()}
            continue
        if args.stop_at:
            # run steps up to and including stop_at
            for name, fn in [("old_outputs", model.step_old_outputs), ("trace", model.step_trace), ("build", model.step_build),
                             ("backup", model.step_backup), ("install", model.step_install), ("new_outputs", model.step_new_outputs),
                             ("gate", model.step_gate), ("upload", model.step_upload)]:
                if not model.done(name):
                    log(f"{m}: {name} …")
                    if not fn():
                        break
                if name == args.stop_at:
                    break
        else:
            model.run_all()
        summary[m] = {k: (v.get("verdict") or v.get("state") or ("ok" if v.get("ok") else "failed")) for k, v in model.state["steps"].items()}
        sp = Path(args.out) / "summary.json"
        merged = json.loads(sp.read_text()) if sp.exists() else {}
        merged.update(summary)
        sp.write_text(json.dumps(merged, indent=1))
    print(json.dumps(summary, indent=1))
    # The exit code tells the truth: a model whose chain stopped (a gate that did not pass, a refused,
    # deferred or failed upload) makes the run fail, so a marker chained on this command cannot say DONE.
    incomplete = [m for m, st in summary.items() if not (st.get("upload") == "ok")]
    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
