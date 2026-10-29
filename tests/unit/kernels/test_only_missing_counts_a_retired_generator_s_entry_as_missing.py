"""`--only-missing` asks for a certification the runtime SERVES: one proven under a retired code generator is not
served (`entry_for_generator`), so it is missing, not present. The Mac, 2026-10-25 02:43: 17 addmm fp32 entries of the
Apple directory held proofs under a retired generator; `--only-missing` counted them present, the runtime served
nothing, and HAT's cells were withheld until `--reprove-generator` re-proved them.

What this test would do if the code were wrong: with `--only-missing` ignoring the generator (the behaviour before),
the first assertion fails — the retired entry reads as coverage. Injection seen red on 2026-10-25, restored green.
"""
from nbxtensor.ops import autotune_certified as C


def _entry(backend):
    proof = {"machine": {"device": {"ordinal": 0, "visible_devices": "0", "name": "V100", "memory_mb": 16384},
                         "clocks_mhz": {"0": {"graphics": 1290, "memory": 877}}},
             "backend": backend}
    return {"config": {"kwargs": {}, "num_warps": 4, "num_stages": 3}, "proof": proof, "excluded": []}


def test_only_missing_asks_for_the_running_generator():
    running = C.proof_backend({"backend": {"name": "cuda", "triton": "3.8.0"}})
    retired = {"k": _entry({"name": "cuda", "triton": "3.6.0"})}
    current = {"k": _entry({"name": "cuda", "triton": "3.8.0"})}
    assert C.coverage_asked(retired, "k", 16, True, False, False, running) is False     # not served: missing
    assert C.coverage_asked(current, "k", 16, True, False, False, running) is True
    assert C.coverage_asked({}, "k", 16, True, False, False, running) is False


def test_without_a_flag_nothing_counts_as_covered_and_the_clock_flag_still_asks_the_clock():
    running = C.proof_backend({"backend": {"name": "cuda", "triton": "3.8.0"}})
    current = {"k": _entry({"name": "cuda", "triton": "3.8.0"})}
    assert C.coverage_asked(current, "k", 16, False, False, False, running) is False    # a full sweep re-proves all
    assert C.coverage_asked(current, "k", 16, False, True, False, running) is True      # clocked: covered
    del current["k"]["proof"]["machine"]["clocks_mhz"]
    assert C.coverage_asked(current, "k", 16, False, True, False, running) is False     # unclocked: re-proven
