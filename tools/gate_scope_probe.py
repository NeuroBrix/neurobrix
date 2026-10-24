#!/usr/bin/env python3
"""The evaluation half of the computed gate scope: runs inside ONE tree (its `src/` first on the path, no
card visible) and evaluates what the static half (tools/gate_scope.py) asks of that tree's own code.

One process per (tree, memory class). It first puts itself in the gate's state for that class — the rack
setup the cell's process runs before any op (data: tools/gate_scope.yml `rack`), on the rack's profile —
then answers jobs read as JSON from stdin:

    argv    the tree's own parser over every cell's argv: {cell: vars(namespace)}
    opmap   the tree's own op map: {key: "module:qualname" of the function it dispatches}
    const   whether a call of the tree's function is a CONSTANT at the gate: called with opaque arguments
            (any use of one beyond reading an attribute off it raises), and again with every argument None;
            constant when both give the same literal without touching an argument. The tree lines it
            executed are returned (the static half checks no reachable code writes what they read), and the
            environment variables it read.
    points  a piece of the tree's code (a function, or a changed region wrapped as one), evaluated at every
            point of a finite domain in the namespace of the module it lives in; per point the value's
            repr or `raise <Type>`, and the probe source lines it executed (the coverage the static half
            requires of a changed region).

Values are specs, built here:
    {"lit": v} | {"tuple": [spec]} | {"list": [spec]} | {"dict": {k: spec}}
    {"member": "pkg.mod:Enum", "name": "x"}
    {"call": "pkg.mod:qual", "args": [...], "kwargs": {...}}
    {"obj": "pkg.mod:Class", "attrs": {name: spec}}     an instance, its __init__ not run
    {"expr": "python expression"}                        evaluated in the probe's module namespace
Every result is `repr` of the value, or `raise <ExceptionType>` — an exception is a result like any other.
"""
from __future__ import annotations

import builtins
import importlib
import json
import os
import sys
import threading

# -------------------------------------------------------------------------------------------------------------
# opaque arguments
# -------------------------------------------------------------------------------------------------------------


class Touched(Exception):
    """An opaque argument was used for more than reading an attribute off it."""


def _touch(name):
    def f(self, *a, **k):
        raise Touched(name)
    return f


class Opaque:
    """An argument the constant probe does not know: attribute reads give a child opaque; every other use
    (truth, comparison, iteration, call, str, arithmetic, isinstance) raises Touched. Hashable by identity
    (a memo keyed on it does not decide anything)."""
    __slots__ = ("_path",)

    def __init__(self, path):
        object.__setattr__(self, "_path", path)

    def __getattr__(self, a):
        if a.startswith("__"):
            raise Touched(a)
        return Opaque(f"{self._path}.{a}")

    def __setattr__(self, a, v):
        raise Touched("setattr")

    def __hash__(self):
        return id(self)


for _d in ("__bool__", "__len__", "__iter__", "__str__", "__repr__", "__format__", "__call__", "__getitem__",
           "__setitem__", "__contains__", "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__", "__add__",
           "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__", "__floordiv__", "__mod__",
           "__index__", "__int__", "__float__", "__fspath__", "__enter__", "__neg__", "__invert__", "__and__",
           "__or__", "__rand__", "__ror__", "__matmul__", "__pow__"):
    setattr(Opaque, _d, _touch(_d))
Opaque.__repr__ = lambda self: f"<opaque {self._path}>"         # read by error messages only


def _has_opaque(x) -> bool:
    if isinstance(x, Opaque):
        return True
    if isinstance(x, (list, tuple, set, frozenset)):
        return any(_has_opaque(v) for v in x)
    if isinstance(x, dict):
        return any(_has_opaque(k) or _has_opaque(v) for k, v in x.items())
    return False


_REAL_ISINSTANCE, _REAL_HASATTR, _REAL_CALLABLE = builtins.isinstance, builtins.hasattr, builtins.callable


def _guarded(real, name):
    def f(obj, *a):
        if type(obj) is Opaque:
            raise Touched(name)
        return real(obj, *a)
    return f


# -------------------------------------------------------------------------------------------------------------
# environment and line recording
# -------------------------------------------------------------------------------------------------------------

class _EnvRecorder(dict):
    def __init__(self, real, seen):
        super().__init__(real)
        self._seen = seen

    def __getitem__(self, k):
        self._seen.add(k)
        return super().__getitem__(k)

    def get(self, k, d=None):
        self._seen.add(k)
        return super().get(k, d)

    def __contains__(self, k):
        self._seen.add(k)
        return super().__contains__(k)


class _Lines:
    """sys.settrace recorder of the tree lines executed, suspended inside the declared rack reader (it reads the
    rack's vendor profile, its own memo included)."""

    def __init__(self, root: str):
        self.root, self.lines, self.suspended = root, set(), 0

    def tracer(self, frame, event, arg):
        if self.suspended:
            return None
        f = frame.f_code.co_filename
        if not f.startswith(self.root):
            return None

        def local(fr, ev, a):
            if ev == "line" and not self.suspended:
                self.lines.add((fr.f_code.co_filename[len(self.root):].lstrip("/"), fr.f_lineno))
            return local
        if event == "call":
            self.lines.add((f[len(self.root):].lstrip("/"), frame.f_lineno))
        return local


# -------------------------------------------------------------------------------------------------------------
# values
# -------------------------------------------------------------------------------------------------------------

def _resolve(qual: str):
    mod, _, name = qual.partition(":")
    obj = importlib.import_module(mod)
    for part in name.split(".") if name else []:
        obj = getattr(obj, part)
    return obj


def build(spec, ns=None):
    if "lit" in spec:
        return spec["lit"]
    if "tuple" in spec:
        return tuple(build(s, ns) for s in spec["tuple"])
    if "list" in spec:
        return [build(s, ns) for s in spec["list"]]
    if "dict" in spec:
        return {k: build(v, ns) for k, v in spec["dict"].items()}
    if "member" in spec:
        enum = _resolve(spec["member"])
        for name in [spec["name"]] + [spec["name"] + x for x in spec.get("suffixes", [])]:
            if hasattr(enum, name):
                return getattr(enum, name)
        raise ValueError(f"{spec['member']} has no member {spec['name']!r}")
    if "call" in spec:
        return _resolve(spec["call"])(*[build(a, ns) for a in spec.get("args", [])],
                                      **{k: build(v, ns) for k, v in spec.get("kwargs", {}).items()})
    if "obj" in spec:
        cls = _resolve(spec["obj"])
        obj = cls.__new__(cls)
        for k, v in spec.get("attrs", {}).items():
            object.__setattr__(obj, k, build(v, ns))
        return obj
    if "expr" in spec:
        # an expression the static half wrote from tools/gate_scope.yml (a declared external domain), never
        # read from a graph or the network
        return eval(spec["expr"], dict(ns or {}, __R=_resolve))
    if "opaque" in spec:
        return Opaque(spec["opaque"])
    raise ValueError(f"unknown value spec {spec!r}")


def show(value) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and type(value).__module__ != "builtins" and hasattr(type(value), "__members__"):
        return f"{type(value).__name__}.{name}"
    return repr(value)


def _literal(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return True
    if isinstance(v, tuple):
        return all(_literal(x) for x in v)
    return False


# -------------------------------------------------------------------------------------------------------------
# the gate's state
# -------------------------------------------------------------------------------------------------------------

class Rack:
    def __init__(self, cfg: dict, hw: str, src_root: str):
        self.cfg, self.hw, self.root = cfg, hw, src_root
        self.lines = None
        reader_mod, _, reader_name = cfg["reader"].partition(":")
        mod = importlib.import_module(reader_mod)
        real = getattr(mod, reader_name)
        values = list(cfg["reader_values"])
        rack = self

        def reader(*args, **kwargs):
            # the declared rack reader answers the rack's own (brand, architecture) for an argument the probe
            # does not know: the gate's process runs on this rack
            args = [values[i] if i < len(values) and type(a) is Opaque else a for i, a in enumerate(args)]
            if rack.lines is not None:
                rack.lines.suspended += 1
            try:
                return real(*args, **kwargs)
            finally:
                if rack.lines is not None:
                    rack.lines.suspended -= 1
        reader.__wrapped__ = real
        setattr(mod, reader_name, reader)
        # every module that bound the reader by name binds the wrapper
        for m in list(sys.modules.values()):
            if getattr(m, "__name__", "").startswith(cfg.get("package", "neurobrix")) and \
                    getattr(m, reader_name, None) is real:
                setattr(m, reader_name, reader)
        self._reader_name, self._real, self._wrapper = reader_name, real, reader
        # the launcher's target from the class's hardware profile, as the census shadow binds it: with no card
        # the driver never answers, and a vendor profile left to it resolves empty and is re-probed at every read
        _resolve(cfg["target_binder"])(hw, None)
        profile = _resolve(cfg["profile_loader"])(hw)
        _resolve(cfg["setup"])(profile)
        self.profile = profile
        vendor = _resolve(cfg["vendor_profile"])()
        if not vendor:
            raise RuntimeError(f"the vendor profile resolved EMPTY behind the door for the hardware profile {hw!r}: "
                               f"every point would re-probe the driver and be answered under no profile")
        self.vendor_profile = vendor.get("_profile")

    def rebind(self):
        for m in list(sys.modules.values()):
            if getattr(m, "__name__", "").startswith(self.cfg.get("package", "neurobrix")) and \
                    getattr(m, self._reader_name, None) is self._real:
                setattr(m, self._reader_name, self._wrapper)


def _call_const(rack: Rack, fn, nparams: int, method: bool, variant: str):
    args = []
    n = nparams
    for i in range(n):
        args.append(Opaque(f"arg{i}") if variant == "opaque" else None)
    env_seen = set()
    real_env = os.environ
    lines = _Lines(rack.root)
    rack.lines = lines
    os.environ = _EnvRecorder(real_env, env_seen)
    builtins.isinstance = _guarded(_REAL_ISINSTANCE, "isinstance")
    builtins.hasattr = _guarded(_REAL_HASATTR, "hasattr")
    builtins.callable = _guarded(_REAL_CALLABLE, "callable")
    sys.settrace(lines.tracer)
    threading.settrace(lines.tracer)
    out = {}
    try:
        v = fn(*args)
        out["literal"] = _literal(v) and not _has_opaque(v)
        out["value"] = v if out["literal"] else None
        out["repr"] = show(v) if not _has_opaque(v) else "<opaque>"
    except Touched as exc:
        out["touched"] = str(exc)
    except (Exception, SystemExit) as exc:       # noqa: BLE001 — a raise is a result, an exit too
        # a CLI command's body exits (`sys.exit(1)` when no daemon runs): uncaught, it ended the whole probe and the
        # scope was refused for every lot whose diff reached the CLI (the Mac's pointer slots, 2026-10-20 19:43)
        out["raised"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    finally:
        sys.settrace(None)
        threading.settrace(None)
        builtins.isinstance, builtins.hasattr, builtins.callable = _REAL_ISINSTANCE, _REAL_HASATTR, _REAL_CALLABLE
        os.environ = real_env
        rack.lines = None
    out["lines"] = sorted(lines.lines)
    out["env"] = sorted(env_seen)
    return out


def job_const(rack: Rack, j: dict) -> dict:
    try:
        fn = _resolve(j["target"])
    except Exception as exc:                     # noqa: BLE001
        return {"error": f"resolve {j['target']}: {type(exc).__name__}: {exc}"[:300]}
    fn = getattr(fn, "__wrapped__", fn) if j.get("unwrap") else fn
    rack.rebind()
    n = j["nparams"]
    a = _call_const(rack, fn, n, j.get("method", False), "opaque")
    out = {"opaque": a}
    if a.get("literal"):
        b = _call_const(rack, fn, n, j.get("method", False), "none")
        out["none"] = b
        # constant: the same literal with every argument None, or None refused (raised) there
        same = (b.get("literal") and b.get("value") == a.get("value") and type(b.get("value")) is type(a.get("value"))) \
            or ("raised" in b) or ("touched" in b)
        out["constant"] = bool(same)
        out["value"] = a["value"]
        out["lines"] = sorted(set(map(tuple, a["lines"])) | set(map(tuple, b.get("lines", []))))
        out["env"] = sorted(set(a["env"]) | set(b.get("env", [])))
    else:
        out["constant"] = False
    return out


def job_points(j: dict) -> dict:
    try:
        ns = dict(vars(importlib.import_module(j["module"])))
    except Exception as exc:                     # noqa: BLE001 — a module that does not import is a result
        return {"error": f"import {j['module']}: {type(exc).__name__}: {exc}"[:300]}
    ns["__R"] = _resolve
    fname = f"<probe {j['id']}>"
    try:
        exec(compile(j["source"], fname, "exec"), ns)
        fn = ns[j["entry"]]
    except Exception as exc:                     # noqa: BLE001
        return {"error": f"compile: {type(exc).__name__}: {exc}"[:300]}
    covered = set()

    def tracer(frame, event, arg):
        if frame.f_code.co_filename != fname:
            return None

        def local(fr, ev, a):
            if ev == "line":
                covered.add(fr.f_lineno)
            return local
        return local
    out = []
    for point in j["points"]:
        try:
            args = {k: build(v, ns) for k, v in point.items()}
        except Exception as exc:                 # noqa: BLE001 — a value the tree cannot build is said
            out.append(f"unbuildable {type(exc).__name__}: {str(exc)[:120]}")
            continue
        if j.get("guard_builtins"):
            # an opaque argument answers no isinstance / hasattr / callable either (a silent False would decide)
            builtins.isinstance = _guarded(_REAL_ISINSTANCE, "isinstance")
            builtins.hasattr = _guarded(_REAL_HASATTR, "hasattr")
            builtins.callable = _guarded(_REAL_CALLABLE, "callable")
        sys.settrace(tracer)
        try:
            v = fn(**args)
            out.append("<opaque>" if _has_opaque(v) else show(v))
        except Exception as exc:                 # noqa: BLE001 — an exception is the probe's result
            out.append(f"raise {type(exc).__name__}")
        finally:
            sys.settrace(None)
            builtins.isinstance, builtins.hasattr, builtins.callable = _REAL_ISINSTANCE, _REAL_HASATTR, \
                _REAL_CALLABLE
    return {"results": out, "covered": sorted(covered)}


def job_argv(j: dict) -> dict:
    parser = _resolve(j["parser"])()
    out = {}
    for cell, argv in j["argvs"].items():
        try:
            ns = parser.parse_args(argv)
        except SystemExit as exc:
            out[cell] = {"__error__": f"argv refused ({exc.code})"}
            continue
        out[cell] = {k: (v if _literal(v) or isinstance(v, (list, dict)) else repr(v)) for k, v in vars(ns).items()
                     if not callable(v)}
        out[cell]["__func__"] = {k: f"{v.__module__}:{v.__qualname__}" for k, v in vars(ns).items() if callable(v)}
    return out


def job_opmap(j: dict) -> dict:
    table = _resolve(j["builder"])() if j.get("builder") else _resolve(j["table"])
    out = {}
    for k, v in table.items():
        f = getattr(v, "__wrapped__", v)
        out[k] = f"{getattr(f, '__module__', '?')}:{getattr(f, '__qualname__', repr(f))}"
    canon = _resolve(j["key"]) if j.get("key") else None
    keys = {}
    for t in j.get("op_types", []):
        bare = t.split("::")[-1]
        keys[t] = canon(bare) if canon else t
    return {"table": out, "keys": keys}


def _setup(req):
    """(root, rack) of a request: the tree's source and its rack, set up once (the engine imported with it)."""
    root = os.path.realpath(req["src"])
    rack = Rack(req["rack"], req["hw"], root) if req.get("rack") else None
    return root, rack


def serve() -> int:
    """`--serve`: one probe process per (tree, class, rack) that imports the engine ONCE and forks a child per request
    (one JSON line on stdin each, the first one setting the tree up) — every job still runs in a process of its own,
    its state gone with it, without paying the engine's import again (a probe process spent nearly all its time in
    that import: c1restlot's scope, ~930 probes in 42 min, 2026-10-23). Each answer is one line on stdout."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        print(json.dumps({"error": "REFUSED: a probe runs with no card visible (CUDA_VISIBLE_DEVICES='')"}), flush=True)
        return 2
    first = json.loads(sys.stdin.readline())
    try:
        _root, rack = _setup(first)
    except Exception as exc:                     # noqa: BLE001 — a rack that cannot be set up is named
        print(MARK_SERVE + json.dumps({"error": f"rack setup: {type(exc).__name__}: {exc}"[:400]}), flush=True)
        return 3
    print(MARK_SERVE + json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:                             # the job's own process
            os.close(r)
            try:
                out = _run_jobs(rack, req["jobs"])
                data = json.dumps(out, default=repr).encode()
            except BaseException as exc:         # noqa: BLE001
                data = json.dumps({"error": f"{type(exc).__name__}: {exc}"[:400]}).encode()
            with os.fdopen(w, "wb") as fh:
                fh.write(data)
            os._exit(0)
        os.close(w)
        with os.fdopen(r, "rb") as fh:
            data = fh.read()
        os.waitpid(pid, 0)
        print(MARK_SERVE + (data.decode() or json.dumps({"error": "the job's process wrote nothing"})), flush=True)
    return 0


MARK_SERVE = "@@GATE_SCOPE_SERVE@@"


def _run_jobs(rack, jobs) -> dict:
    out = {}
    for j in jobs:
        k = j["kind"]
        try:
            if k == "const":
                out[j["id"]] = job_const(rack, j)
            elif k == "points":
                out[j["id"]] = job_points(j)
            elif k == "argv":
                out[j["id"]] = job_argv(j)
            elif k == "opmap":
                out[j["id"]] = job_opmap(j)
            else:
                out[j["id"]] = {"error": f"unknown job kind {k!r}"}
        except Exception as exc:                 # noqa: BLE001
            out[j["id"]] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return out


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        return serve()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        print(json.dumps({"error": "REFUSED: a probe runs with no card visible (CUDA_VISIBLE_DEVICES='')"}))
        return 2
    req = json.load(sys.stdin)
    try:
        _root, rack = _setup(req)
    except Exception as exc:                     # noqa: BLE001 — a rack that cannot be set up is named
        print(json.dumps({"error": f"rack setup: {type(exc).__name__}: {exc}"[:400]}))
        return 3
    out = _run_jobs(rack, req["jobs"])
    sys.stdout.flush()
    print("\n@@GATE_SCOPE_PROBE@@" + json.dumps(out, default=repr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
