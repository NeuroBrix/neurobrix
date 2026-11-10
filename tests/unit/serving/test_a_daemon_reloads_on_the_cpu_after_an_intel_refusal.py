"""The serving daemon and an Intel GPU's refusal at run time (core/prism/intel_refusal.py; the Dell's review 2026-11-10
16:0x): at load or at a request, a refusal of a chip the daemon's own profile plans is said by name, recorded for that
device, and a NEW engine is loaded ONCE on the profile detection then gives; the refused engine is released only once
the new one is loaded, so a reload that fails leaves the daemon as it was. A second refusal, a result that is no verdict
on the chip, any other error, or a profile the user gave (--hardware) changes nothing. Card-less: a stand-in engine.
What this test would do if the code were wrong: a reload loop, an explicit profile replaced, the wrong profile
deleted, the old engine released before the new one loads (a daemon with no engine), a transient error recorded (red)."""
import pytest

from nbxtensor.backend.level_zero import IntelDeviceError
from neurobrix.core.prism import autodetect, intel_refusal
from neurobrix.serving.server import ServingDaemon

PROFILE = "devices:\n- {index: 0, brand: intel, model: Arc A770, pci_id: '8086:56a0'}\n"


class Engine:
    built, fail_next_load = [], False

    def __init__(self, model_name, hardware_id, mode="triton"):
        self.model_name, self.hardware_id, self.mode, self.loaded, self.unloaded = model_name, hardware_id, mode, False, False
        Engine.built.append(self)

    def load(self):
        if Engine.fail_next_load:
            raise RuntimeError("no plan fits the CPU")
        self.loaded = True

    def unload(self):
        self.unloaded = True


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    monkeypatch.setattr(autodetect, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr(autodetect, "DEFAULT_PROFILE_PATH", tmp_path / "default.yml")
    monkeypatch.setattr(autodetect, "get_or_create_default_profile", lambda: "default-cpu")   # what detection gives next
    monkeypatch.setattr(intel_refusal, "_versions", lambda: {"driver": "1.6.31294", "nbxtensor": "0.1.0"})
    (tmp_path / "default-0a1b.yml").write_text(PROFILE)
    (tmp_path / "default-cpu.yml").write_text("devices: []\n")
    Engine.built.clear()
    Engine.fail_next_load = False

    def make(given=None):
        return ServingDaemon("m", given, mode="triton", engine=Engine("m", given or "default-0a1b"), idle_timeout=60)
    return make, tmp_path


def _refusal(rc=0x70000004):
    try:
        try:
            raise IntelDeviceError("zeModuleCreate failed", "zeModuleCreate", rc, "error: unsupported", device=0)
        except IntelDeviceError as e:
            raise RuntimeError("Failed at op aten.add::3") from e
    except RuntimeError as outer:
        return outer


def test_the_daemon_says_records_and_loads_once_on_the_plan_detection_gives(daemon):
    make, tmp = daemon
    d = make()
    first = d._engine
    said = d._reload_after_intel_refusal(_refusal())
    assert said and "Arc A770 (8086:56a0)" in said and "zeModuleCreate" in said and intel_refusal.LABEL in said
    assert d._engine is not first and d._engine.loaded and d._engine.hardware_id == "default-cpu" and first.unloaded
    assert intel_refusal.recorded("8086:56a0")
    assert not (tmp / "default-0a1b.yml").exists() and (tmp / "default-cpu.yml").exists()   # the one that planned it
    assert d._reload_after_intel_refusal(_refusal()) is None and len(Engine.built) == 2      # once: never a loop


def test_a_reload_that_fails_leaves_the_daemon_with_its_engine(daemon):
    make, _ = daemon
    d = make()
    first = d._engine
    Engine.fail_next_load = True
    with pytest.raises(RuntimeError, match="no plan fits the CPU"):
        d._reload_after_intel_refusal(_refusal())
    assert d._engine is first and not first.unloaded                 # never a daemon without an engine
    Engine.fail_next_load = False
    assert d._reload_after_intel_refusal(_refusal()) is None          # and never a second attempt


@pytest.mark.parametrize("case", ["given profile", "another error", "transient result"])
def test_nothing_is_replaced_or_recorded_otherwise(daemon, case):
    make, tmp = daemon
    d = make("my-profile" if case == "given profile" else None)
    exc = {"given profile": _refusal(), "another error": RuntimeError("out of memory"),
           "transient result": _refusal(rc=0x70000001)}[case]
    assert d._reload_after_intel_refusal(exc) is None and len(Engine.built) == 1
    assert not (tmp / "intel_runtime_refusals").exists() and (tmp / "default-0a1b.yml").exists()
