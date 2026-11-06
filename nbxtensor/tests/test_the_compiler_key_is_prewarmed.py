"""`backend.launcher.prewarm_compiler` computes Triton's cache key (`triton_key`, the sha256 of libtriton and Triton's sources,
1.3 s on the V100 rack) in a daemon thread, so a run's first compile finds it computed. Measured at the entry point, Kokoro-82M
--triton on a V100-16GB alone, three runs each (2026-11-06): 10.47 s mean without, 9.12 s with, the same output sha. What this
test does if the code were wrong: the key not computed, or computed on the caller's thread (blocking the start it exists to
overlap) — red."""
import threading

from nbxtensor.backend import launcher


def test_the_key_is_computed_in_a_background_thread(monkeypatch):
    import triton.runtime.cache as cache
    seen = []
    done = threading.Event()

    def key():
        seen.append(threading.current_thread().name)
        done.set()
        return "k"
    monkeypatch.setattr(cache, "triton_key", key)
    launcher.prewarm_compiler()
    assert done.wait(10), "the key was never computed"
    assert seen == ["nbxtensor-triton-key"], seen
