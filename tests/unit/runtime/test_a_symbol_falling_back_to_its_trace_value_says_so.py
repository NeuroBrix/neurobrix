"""A symbol no fed input binds used to resolve to its TRACE value through a
`logger.warning` nobody read — a dimension frozen at the trace, the class the
doctrine forbids, silent by construction (Qwen3-Omni's fresh container,
2026-09-16: `view [-1, 1, 23]` on 639 elements, s1 = 23). Now the fallback is
said once per symbol on stdout, and NBX_STRICT_SYMBOLS=1 refuses it by name.
Injection: the print removed → the first test read no FALLBACK line — RED."""
import pytest

from neurobrix.core.runtime.shape_resolver import SymbolicShapeResolver, ShapeResolutionError

CTX = {"symbols": {"s1": {"name": "seq_len", "trace_value": 23, "source": "input::inputs_embeds::dim_1",
                         "constraints": {"min": 1}}}}


def test_the_fallback_is_said_once_on_stdout(capsys, monkeypatch):
    monkeypatch.delenv("NBX_STRICT_SYMBOLS", raising=False)
    r = SymbolicShapeResolver(CTX)
    assert r.resolve("s1") == 23 and r.resolve(["s1", 4]) == [23, 4]
    out = capsys.readouterr().out
    assert out.count("[SymShape] FALLBACK: symbol s1") == 1 and "input::inputs_embeds::dim_1" in out


def test_strict_refuses_by_name(monkeypatch):
    monkeypatch.setenv("NBX_STRICT_SYMBOLS", "1")
    r = SymbolicShapeResolver(CTX)
    with pytest.raises(ShapeResolutionError, match="s1.*inputs_embeds.*NBX_STRICT_SYMBOLS"):
        r.resolve("s1")
