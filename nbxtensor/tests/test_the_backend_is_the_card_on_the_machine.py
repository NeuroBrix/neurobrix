"""The runtime-library backend is the one whose card is on the machine's PCI bus (`runtime._on_this_machine`).

nbxtensor 0.0.2 depends on the CUDA runtime wheel on Linux, and a runtime library loads without its card: on an AMD or
Intel machine `gpu_backend()` named "cuda" and HIP / Level Zero were never tried (the supervisor's reading of the 0.0.2
metadata, 2026-11-06 18:25). What this test does if the code were wrong: "cuda" kept on a bus with only an AMD card, or
a backend dropped where the bus cannot be read — red."""
from nbxtensor.backend import install_help, runtime


def test_only_the_backends_of_the_cards_present_are_tried(monkeypatch):
    table = list(runtime._GPU_BACKENDS.items())
    monkeypatch.setattr(install_help, "pci_gpus", lambda: [("amd", 0x1002, 0x744C)])
    assert [n for n, _b in runtime._on_this_machine(table)] == ["hip"]
    monkeypatch.setattr(install_help, "pci_gpus", lambda: [("intel", 0x8086, 0xE20B)])
    assert runtime._on_this_machine(table) == []                     # Level Zero is probed for its device after
    monkeypatch.setattr(install_help, "pci_gpus", lambda: [("nvidia", 0x10DE, 0x1DB1)])
    assert [n for n, _b in runtime._on_this_machine(table)] == ["cuda"]
    monkeypatch.setattr(install_help, "pci_gpus", lambda: [])      # no readable bus (macOS, a sandbox): as before
    assert [n for n, _b in runtime._on_this_machine(table)] == ["cuda", "hip"]
