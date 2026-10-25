"""A hardware profile without a `measured:` block (an auto-generated one, the Mac's) takes the device's and the host's
measured facts from the TRACKED vendor file of its DEVICE (the chip file its compute capability names, else its architecture's, config/vendors/<brand>/*.yml — the kernels' own resolution); a profile
that carries its own keeps them; a vendor file without the block gives None — never an invented figure.

What this test would do if the code were wrong: a loader that reads the profile only leaves `measured` None for the
test profile and the first assertion fails. Injection seen red on 2026-10-25, restored green.
"""
from neurobrix.core.prism import loader as L

V = {"measured": {"source": "test", "matmul_tflops": {"float16": 1.0, "float32": 0.5}, "memory_gbps": 10.0, "h2d_gbps": 1.0},
     "host_measured": {"source": "test", "host_matmul_gflops": 100.0, "host_memory_gbps": 20.0}}


def test_the_vendor_file_supplies_the_blocks_a_profile_lacks(monkeypatch):
    import neurobrix.core.config.loader as CL
    monkeypatch.setattr(CL, "vendor_config_for_device", lambda vendor, arch, cc: dict(V))
    p = L.load_profile("test_profile")                      # the committed test profile carries no measured block
    assert p.devices[0].measured == V["measured"] and p.host_measured == V["host_measured"]


def test_a_profile_s_own_block_wins_and_a_vendor_without_one_gives_none(monkeypatch):
    import neurobrix.core.config.loader as CL
    monkeypatch.setattr(CL, "vendor_config_for_device", lambda vendor, arch, cc: {"measured": {"source": "vendor"}})
    p = L.load_profile("v100-16g")                          # carries its own measured block (the rack's, 2026-10-24)
    assert p.devices[0].measured["source"] != "vendor" and "matmul_tflops" in p.devices[0].measured
    monkeypatch.setattr(CL, "vendor_config_for_device", lambda vendor, arch, cc: {})
    assert L.load_profile("test_profile").devices[0].measured is None
    assert L.load_profile("test_profile").host_measured is None


def test_the_chip_file_is_read_before_the_family_file(tmp_path, monkeypatch):
    """The Mac, 2026-10-25 17:20: the M4 Pro's facts live in apple/apple_m4_pro.yml (compute_capability apple-m4-pro),
    not in the family file apple_silicon.yml the architecture names — the estimate found none. The loader now resolves
    the device's vendor file by the kernels' own rule (the chip file first)."""
    import yaml
    from neurobrix.core.config import loader as CL
    root = tmp_path / "vendors" / "apple"; root.mkdir(parents=True)
    (root / "apple_silicon.yml").write_text(yaml.safe_dump({"architecture": "apple_silicon"}))
    (root / "apple_m4_pro.yml").write_text(yaml.safe_dump({"architecture": "apple_silicon", "compute_capability": "apple-m4-pro",
                                                          "measured": {"source": "test", "matmul_tflops": {"float16": 6.0}, "memory_gbps": 224, "h2d_gbps": 58}}))
    monkeypatch.setattr(CL, "CONFIG_ROOT", tmp_path)
    from neurobrix.core.prism.loader import _vendor_block
    dev = {"brand": "apple", "architecture": "apple_silicon", "compute_capability": "apple-m4-pro"}
    assert (_vendor_block(dev, "measured") or {}).get("memory_gbps") == 224
    family = {"brand": "apple", "architecture": "apple_silicon", "compute_capability": "apple-m9"}
    assert _vendor_block(family, "measured") is None                       # the family file carries none: nothing invented
