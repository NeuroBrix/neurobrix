"""The serving daemon and an Intel GPU's refusal at run time (core/prism/intel_refusal.py; the Dell + the Mac 2026-11-10):
at load or at a request, nbxtensor's IntelDeviceError is said by name, recorded, and the daemon builds its engine again
on the profile detection then gives (the CPU's) and loads it ONCE; a second refusal, any other error, or a profile the
user gave (--hardware) is not replaced. Card-less: a stand-in engine whose first load raises the typed error.
What this test would do if the code were wrong: a reload loop, an explicit profile replaced, an ordinary failure
treated as Intel's, a refused chip never recorded (red)."""
import pytest

from nbxtensor.backend.level_zero import IntelDeviceError
from neurobrix.core.prism import autodetect, intel_refusal
from neurobrix.serving.server import ServingDaemon


class Engine:
    built = []

    def __init__(self, model_name, hardware_id, mode="triton"):
        self.model_name, self.hardware_id, self.mode, self.loaded, self.unloaded = model_name, hardware_id, mode, False, False
        Engine.built.append(self)

    def load(self):
        self.loaded = True

    def unload(self):
        self.unloaded = True


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    monkeypatch.setattr(autodetect, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr(autodetect, "DEFAULT_PROFILE_PATH", tmp_path / "default.yml")
    monkeypatch.setattr(autodetect, "_visible_set_tag", lambda: "abc")
    monkeypatch.setattr(autodetect, "get_or_create_default_profile", lambda: "default-cpu")
    monkeypatch.setattr(intel_refusal, "_versions", lambda: {"driver": "1.6.31294", "nbxtensor": "0.1.0"})
    (tmp_path / "default-abc.yml").write_text("devices:\n- {brand: intel, model: Arc, pci_id: '8086:7d55'}\n")
    Engine.built.clear()

    def make(hardware_id=None):
        return ServingDaemon("m", hardware_id, mode="triton", engine=Engine("m", hardware_id or "default-abc"), idle_timeout=60)
    return make, tmp_path


def _refusal():
    try:
        try:
            raise IntelDeviceError("zeModuleCreate failed", "zeModuleCreate", 0x70000004, "error: unsupported")
        except IntelDeviceError as e:
            raise RuntimeError("Failed at op aten.add::3") from e
    except RuntimeError as outer:
        return outer


def test_the_daemon_says_records_and_loads_once_on_the_plan_detection_gives(daemon):
    make, tmp = daemon
    d = make()
    d._auto_hardware = True                                  # it resolved the machine's profile itself
    first = d._engine
    said = d._reload_after_intel_refusal(_refusal())
    assert said and "8086:7d55" in said and "zeModuleCreate" in said and intel_refusal.LABEL in said
    assert first.unloaded and d._engine is not first and d._engine.loaded and d._engine.hardware_id == "default-cpu"
    assert intel_refusal.recorded("8086:7d55") and not (tmp / "default-abc.yml").exists()
    assert d._reload_after_intel_refusal(_refusal()) is None and len(Engine.built) == 2     # once: never a loop


def test_an_explicit_profile_or_another_error_is_not_replaced(daemon):
    make, tmp = daemon
    d = make("my-profile")
    assert d._reload_after_intel_refusal(_refusal()) is None and len(Engine.built) == 1
    d2 = make()
    d2._auto_hardware = True
    assert d2._reload_after_intel_refusal(RuntimeError("out of memory")) is None
    assert not (tmp / "intel_runtime_refusals.json").exists()
