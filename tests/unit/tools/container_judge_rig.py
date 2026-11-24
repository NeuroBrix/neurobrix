"""A run root for the container judge's tests: the builder's three files, a cache holding the containers they name,
and a stand-in for `neurobrix run` that writes each arm's output (no model, no card)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from tests.unit.workshop_configuration import the_workshop_configuration  # noqa: E402
with the_workshop_configuration():
    import container_judge as J  # noqa: E402

OLD_AT, NEW_AT = "2026-06-09T10:00:00", "2026-11-24T08:00:00"
NBX_BYTES = b"a staged container"
NBX_SHA = hashlib.sha256(NBX_BYTES).hexdigest()
MODEL = "swin2SR-x4"
GRAPH_DIFF = {"annotation_changes": 3, "arg_witnessed": 2, "beyond_annotation": 0, "corrupted_before": 2,
              "corrupted_after": 0, "pruned_dead_ops": 0, "components": {}, "topology": [], "topology_additions": []}


class Rig:
    """`outputs[(tag, arm)]` is the bytes the stand-in run writes for that arm; `calls` holds each run's argv/env."""

    def __init__(self, tmp_path: Path, monkeypatch, family: str = "image", request=None):
        self.out = tmp_path / "work"
        self.cache = tmp_path / "cache"
        self.builder = self.out / MODEL / "builder"
        self.judge_dir = self.out / MODEL / "judge"
        self.builder.mkdir(parents=True)
        self.family = family
        self.request = list(request or ["--prompt", "a red apple", "--seed", "42"])
        self.outputs = {("old", "sequential"): b"A", ("old", "triton"): b"A",
                        ("new", "sequential"): b"A", ("new", "triton"): b"A"}
        self.calls = []
        self.gate_calls = []
        monkeypatch.setattr(J, "CACHE", self.cache)
        monkeypatch.setattr(J.C, "CACHE", self.cache)
        monkeypatch.setattr(J.C, "request_args", lambda name, fam, extra: list(self.request) + list(extra))
        monkeypatch.setattr(J.C, "gate", self._gate)
        monkeypatch.setattr(J.one_name, "cli_name", lambda name: name)
        monkeypatch.setattr(J, "run", self._run)
        self.install(OLD_AT)
        self.write("previous.json", {"container": "installed", "cache_path": str(self.cache / MODEL),
                                     "created_at": OLD_AT})

    def install(self, created_at: str) -> None:
        """The cache entry, as the builder's install leaves it."""
        d = self.cache / MODEL
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps({"family": self.family, "created_at": created_at}))

    def build(self, created_at: str = NEW_AT, graph_diff=None) -> None:
        """What the builder's `build` step leaves: the new container installed and staged (`self.nbx`), build.json
        naming the staged file with its sha256 and byte count, graph_diff.json."""
        self.install(created_at)
        self.nbx = self.out.parent / "staged" / f"{MODEL}.nbx"
        self.nbx.parent.mkdir(parents=True, exist_ok=True)
        self.nbx.write_bytes(NBX_BYTES)
        self.write("build.json", {"model": MODEL, "family": self.family, "installed_name": MODEL,
                                  "cache_path": str(self.cache / MODEL), "created_at": created_at,
                                  "nbx_path": str(self.nbx), "sha256": NBX_SHA, "bytes": len(NBX_BYTES)})
        self.write("graph_diff.json", dict(GRAPH_DIFF if graph_diff is None else graph_diff))

    def write(self, name: str, obj) -> None:
        (self.builder / name).write_text(json.dumps(obj))

    def builder_files(self) -> dict:
        return {p.name: p.read_bytes() for p in sorted(self.builder.iterdir())}

    def _run(self, cmd, env, logpath, timeout):
        outp = Path(cmd[cmd.index("--output") + 1])
        tag, arm = outp.stem.split("_", 1)
        self.calls.append({"cmd": [str(c) for c in cmd], "env": env, "tag": tag, "arm": arm})
        logpath.write_text("$ run\n")
        data = self.outputs.get((tag, arm))
        if data is None:
            logpath.write_text("$ run\nRuntimeError: no output\n")
            return 1
        outp.write_bytes(data)
        return 0

    def _gate(self, a, b):
        self.gate_calls.append((Path(a).name, Path(b).name))
        return {"kind": "image", "pass": True, "psnr_db": 40.0, "identical": False}

    def args(self, *argv):
        return J.parser().parse_args(["--out", str(self.out), "--model", MODEL, *argv])

    def main(self, *argv) -> int:
        return J.main(["--out", str(self.out), "--model", MODEL, *argv])

    def state(self) -> dict:
        return json.loads((self.judge_dir / "state.json").read_text())

    def verdict(self) -> dict:
        return json.loads((self.judge_dir / "verdict.json").read_text())
