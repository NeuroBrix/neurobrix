"""A hardware profile without a `measured:` block (an auto-generated one, the Mac's) takes the device's and the host's
measured facts from the TRACKED vendor file of its architecture (config/vendors/<brand>/<architecture>.yml); a profile
that carries its own keeps them; a vendor file without the block gives None — never an invented figure.

What this test would do if the code were wrong: a loader that reads the profile only leaves `measured` None for the
test profile and the first assertion fails. Injection seen red on 2026-10-25, restored green.
"""
from neurobrix.core.prism import loader as L

V = {"measured": {"source": "test", "matmul_tflops": {"float16": 1.0, "float32": 0.5}, "memory_gbps": 10.0, "h2d_gbps": 1.0},
     "host_measured": {"source": "test", "host_matmul_gflops": 100.0, "host_memory_gbps": 20.0}}


def test_the_vendor_file_supplies_the_blocks_a_profile_lacks(monkeypatch):
    import neurobrix.core.config.loader as CL
    monkeypatch.setattr(CL, "get_vendor_config", lambda vendor, arch: dict(V))
    p = L.load_profile("test_profile")                      # the committed test profile carries no measured block
    assert p.devices[0].measured == V["measured"] and p.host_measured == V["host_measured"]


def test_a_profile_s_own_block_wins_and_a_vendor_without_one_gives_none(monkeypatch):
    import neurobrix.core.config.loader as CL
    monkeypatch.setattr(CL, "get_vendor_config", lambda vendor, arch: {"measured": {"source": "vendor"}})
    p = L.load_profile("v100-16g")                          # carries its own measured block (the rack's, 2026-10-24)
    assert p.devices[0].measured["source"] != "vendor" and "matmul_tflops" in p.devices[0].measured
    monkeypatch.setattr(CL, "get_vendor_config", lambda vendor, arch: {})
    assert L.load_profile("test_profile").devices[0].measured is None
    assert L.load_profile("test_profile").host_measured is None
