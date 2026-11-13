"""Prism plans a serving session's prefill chunk on the tables of the decoder chip's VENDOR PROFILE, and a unified
card's host entry (the same memory pool) never refuses it.

On the Mac (2026-11-13), `--explain-plan` said "prefill chunk one per turn: the decoder's cache lives on cpu, which
certifies no kernel time", and a served chat swept attention keys at run time on a clean install of 0.6.14. There were
two causes in `PrismSolver._prefill_chunk`:
- `_kv_devices` adds the host beside a unified card, for the memory accounting: one pool. The chunk then refused on
  that host entry, as if the cache lived on a second device.
- The chunk record and the certified times were read under the device's `architecture` field ("apple_silicon").
  The chip's tables live under its vendor profile ("apple_m4_pro"), the file its reported arch string resolves to.
  On a V100 the two names coincide (`volta`), so this never showed on the rack.

What this test does if the code is wrong:
1. the vendor profile of a device: a V100 does not resolve to `volta`, or the M4 Pro to `apple_m4_pro` (red);
2. TinyLlama planned on the M4 Pro profile with a chunk record under apple/apple_m4_pro/18g: no chunk, or a refusal
   naming cpu (red before the fix);
3. a discrete card whose cache also lists the host (a real host tier) still refuses by name (red if the fix skipped
   every host entry).
"""
import json

from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, V100_16GB, container_root, pin_no_calibration,
                                              profile)

MODEL = "TinyLlama-1.1B-Chat-v1.0"


def _record(model_root, container_sha):
    cands = [{"rows": C, "graph": [["k.mm", f"({C},)", "fp32", 2]],
              "attention": [[tk, "k.sm", f"({C}, {tk})", "fp32", 1] for tk in range(C, 257, 64)]} for C in (64, 128)]
    return {"candidates": cands, "model": model_root.name, "mode": "triton", "container": container_sha}


def _plan(monkeypatch, tmp_path, machine, vendor, table, cls, kv_devices=None):
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    from neurobrix.core.prism import PrismSolver, prefill_chunk as PC
    from neurobrix.kernels import census_table as T
    from neurobrix.nbx import NBXContainer
    pin_no_calibration(monkeypatch)
    if vendor == "nvidia":      # a discrete card is read by its driver: the test plans one, on any host, card-less too
        from neurobrix.core.prism import solver as solver_mod
        from neurobrix.core.prism.memory_budget import DeviceReading
        monkeypatch.setattr(solver_mod, "read_device_sharing", lambda i: DeviceReading(
            kind="device", capacity_mb=16151, free_mb=15845, own_context_mb=306, measured=True,
            source="injected: a quiet card (the test plans; no card is read)"))
    root = container_root(MODEL)
    monkeypatch.setattr(T, "ROOT", tmp_path)
    path = T.table_path(vendor, table, cls)
    path.parent.mkdir(parents=True)
    T.chunks_path(path).write_text(json.dumps(_record(root, T.container_sha(root))) + "\n")
    asked = []
    monkeypatch.setattr(PC, "certified_times",
                        lambda v, p, c: asked.append((v, p, c)) or (lambda kernel, dtype, key: 1.0))
    if kv_devices is not None:
        monkeypatch.setattr(PrismSolver, "_kv_devices", lambda self, allocs, prof=None: set(kv_devices))
    c = NBXContainer.load(str(root))
    man = c.get_manifest() or {}
    args = create_parser().parse_args(["run", "--model", MODEL, "--prompt", "hi", "--max-tokens", "32", "--triton"])
    kv = PrismSolver().solve(c, profile(machine), request_input_config(args, man, man.get("family"), c.cache_path),
                             mode="triton").kv_cache_plan
    return kv, asked


def test_a_device_resolves_to_its_vendor_profile():
    from neurobrix.core.config.loader import vendor_profile_for_device
    assert vendor_profile_for_device("nvidia", "volta", "7.0", "Tesla V100-SXM2-16GB") == "volta"
    assert vendor_profile_for_device("apple", "apple_silicon", "0.0", "Apple M4 Pro") == "apple_m4_pro"


def test_the_m4_pro_plans_a_chunk_on_its_chip_s_tables(monkeypatch, tmp_path):
    kv, asked = _plan(monkeypatch, tmp_path, APPLE_M4_PRO, "apple", "apple_m4_pro", 18)
    assert kv.prefill_chunk in (64, 128), kv.prefill_chunk_reason
    assert "cpu" not in kv.prefill_chunk_reason
    assert asked == [("apple", "apple_m4_pro", 18)]


def test_a_discrete_card_with_a_host_tier_still_refuses_by_name(monkeypatch, tmp_path):
    kv, _ = _plan(monkeypatch, tmp_path, V100_16GB, "nvidia", "volta", 16, kv_devices={"cuda:0", "cpu"})
    assert kv.prefill_chunk is None
    assert "the decoder's cache lives on cpu" in kv.prefill_chunk_reason
