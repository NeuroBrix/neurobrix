"""A vendor file is parsed once per process, not at every per-op read (2026-10-26: `kernels/wrappers._arch_param`, read
by every attention launch, re-parsed volta.yml each time — whisper-large's Triton decode took 4.9 s a token instead of
0.13 s, every Triton model ran 2-20x slower since db94d33d). What this test would do if the code were wrong: a
thousand per-op reads parse the file a thousand times — RED; a caller altering what it was handed changes what the
next caller reads — RED."""
import types

import yaml

from neurobrix.core.config import loader
from neurobrix.kernels import wrappers as w


def _volta_profile():
    dev = types.SimpleNamespace(brand="nvidia", architecture="volta")
    return types.SimpleNamespace(devices=[dev])


def test_a_thousand_per_op_reads_parse_the_vendor_file_once(monkeypatch):
    parses = []
    real = yaml.safe_load
    monkeypatch.setattr(loader.yaml, "safe_load", lambda f: (parses.append(1), real(f))[1])
    if hasattr(loader, "_parsed_vendor_file"):
        loader._parsed_vendor_file.cache_clear()
    monkeypatch.setattr(w, "get_hardware_profile", _volta_profile)
    values = {w._arch_memory_param("sdpa_math_max_chunks", None) for _ in range(1000)}
    assert len(values) == 1 and None not in values, values          # the file's value, read, not the default
    assert len(parses) <= 1, f"{len(parses)} parses for 1000 reads"


def test_what_a_caller_is_handed_cannot_alter_what_the_next_reads():
    first = loader.get_vendor_config("nvidia", "volta")
    first["memory"]["sdpa_math_max_chunks"] = -1
    first.clear()
    again = loader.get_vendor_config("nvidia", "volta")
    assert again and again["memory"]["sdpa_math_max_chunks"] != -1
