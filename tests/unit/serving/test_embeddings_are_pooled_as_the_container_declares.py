"""/v1/embeddings and the embedding family (point 9b, route B validated by the owner 2026-11-09 21:25): the vectors are
the flow's hidden states pooled as the CONTAINER declares (runtime/defaults.json `embedding`), on a fake graph — no
catalogue model is an embedding model yet, and no embedding is derived from a chat model.

What this test would do if the code were wrong: a pooling the engine chose (red: the declaration refused when absent), a
dimension not the container's (red), base64 not the float vector's bytes (red), `dimensions` truncating silently (red),
token-id input accepted (red), a chat family served embeddings (red), a torch tensor (compiled engine) misread (red)."""
from __future__ import annotations

import base64
import struct
import types

import numpy as np
import pytest

from nbxtensor.ops.embedding_pool import l2_normalize, pool_hidden
from neurobrix.core.runtime import output_dispatch as D

HIDDEN = np.random.default_rng(5).standard_normal((1, 4, 3)).astype(np.float32)
MASK = np.array([[1, 1, 1, 0]], dtype=np.int64)


def _executor(decl):
    return types.SimpleNamespace(pkg=types.SimpleNamespace(defaults={"embedding": decl} if decl is not None else {}))


DECL = {"output": "encoder.last_hidden_state", "mask": "global.attention_mask", "pooling": "mean", "normalize": True,
        "dimension": 3}


@pytest.mark.parametrize("pooling", ["mean", "cls", "last"])
def test_the_container_s_pooling_is_applied(pooling):
    out = D.extract_embeddings({DECL["output"]: HIDDEN, DECL["mask"]: MASK}, _executor({**DECL, "pooling": pooling}))
    assert np.allclose(out["embeddings"], l2_normalize(pool_hidden(HIDDEN, MASK, pooling)))
    assert out["dimension"] == 3 and out["tokens"] == 3


def test_a_torch_hidden_state_is_read_as_the_compiled_engine_leaves_it():
    torch = pytest.importorskip("torch")
    out = D.extract_embeddings({DECL["output"]: torch.from_numpy(HIDDEN).to(torch.bfloat16),
                                DECL["mask"]: torch.from_numpy(MASK)}, _executor(DECL))
    want = l2_normalize(pool_hidden(torch.from_numpy(HIDDEN).to(torch.bfloat16).float().numpy(), MASK, "mean"))
    assert np.allclose(out["embeddings"], want)


@pytest.mark.parametrize("decl, outputs, match", [
    (None, {DECL["output"]: HIDDEN}, "declares no"),
    ({**DECL, "dimension": 4}, {DECL["output"]: HIDDEN, DECL["mask"]: MASK}, "not \\[B, T, 4\\]"),
    (DECL, {"other": HIDDEN}, "left no"),
])
def test_refused_by_name(decl, outputs, match):
    with pytest.raises(RuntimeError, match=match):
        D.extract_embeddings(outputs, _executor(decl))


def _door(monkeypatch, family="embedding", result=None):
    from neurobrix.serving.openai_api import routes
    engine = types.SimpleNamespace(family=family)
    daemon = types.SimpleNamespace(_engine=engine, served_name=lambda: "org/emb",
                                   _call_engine=lambda method, params: dict(result))
    monkeypatch.setattr(routes, "_run", lambda door, handler, method, work, deliver=None: (None, work()))
    return routes, types.SimpleNamespace(daemon=daemon)


RESULT = {"embeddings": [[0.6, 0.8, 0.0]], "dimension": 3, "tokens": 2}


def test_the_route_answers_in_openai_s_shape_and_base64_is_the_float_bytes(monkeypatch):
    routes, door = _door(monkeypatch, result=RESULT)
    out = routes.embeddings(door, None, {"model": "org/emb", "input": ["a", "b"]})
    assert out["object"] == "list" and [d["index"] for d in out["data"]] == [0, 1]
    assert out["data"][0] == {"object": "embedding", "index": 0, "embedding": [0.6, 0.8, 0.0]}
    assert out["usage"] == {"prompt_tokens": 4, "total_tokens": 4}
    b64 = routes.embeddings(door, None, {"model": "org/emb", "input": "a", "encoding_format": "base64"})
    raw = base64.b64decode(b64["data"][0]["embedding"])
    assert np.allclose(struct.unpack("<3f", raw), [0.6, 0.8, 0.0])


@pytest.mark.parametrize("body, param", [({"input": [1, 2, 3]}, "input"), ({"input": ""}, "input"),
                                         ({"input": "a", "dimensions": 2}, "dimensions"),
                                         ({"input": "a", "encoding_format": "int8"}, "encoding_format")])
def test_the_route_refuses_by_name(monkeypatch, body, param):
    from neurobrix.serving.openai_api import errors
    routes, door = _door(monkeypatch, result=RESULT)
    with pytest.raises(errors.ApiError) as e:
        routes.embeddings(door, None, {"model": "org/emb", **body})
    assert e.value.status == 400 and getattr(e.value, "param", None) == param


def test_a_chat_family_is_not_served_embeddings(monkeypatch):
    from neurobrix.serving.openai_api import errors
    routes, door = _door(monkeypatch, family="llm", result=RESULT)
    with pytest.raises(errors.ApiError) as e:
        routes.embeddings(door, None, {"model": "org/emb", "input": "a"})
    assert "declares no embedding output" in str(e.value)


def test_a_run_against_the_daemon_writes_the_vectors_where_it_asked(tmp_path):
    """`neurobrix run --output f.json` against a daemon serving an embedding model: the daemon writes the answer there
    (a result without `outputs` was never written: the warm run printed nothing). Red if no file."""
    import json as _json
    from neurobrix.serving.server import ServingDaemon
    d = object.__new__(ServingDaemon)
    d._engine = types.SimpleNamespace(family="embedding", generate=lambda **kw: dict(RESULT))
    out = d._call_engine("generate", {"prompt": "a", "output_path": str(tmp_path / "v.json")})
    assert out["output_path"] == str(tmp_path / "v.json")
    assert _json.loads((tmp_path / "v.json").read_text())["embeddings"] == RESULT["embeddings"]
