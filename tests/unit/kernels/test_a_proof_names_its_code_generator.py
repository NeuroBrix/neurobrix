"""A certified setting is proven for ONE code generator: the proof carries the
Triton version (and the backend name when not cuda), `autotune status` counts
the directory's proofs by generator, and a re-proof under a new Triton is a
planned cost the document names (owner, 2026-09-16: 9 655 entries proven on
3.6.0 are not proven on 3.8). Injection: `proof_backend` returning None for
every proof → the status counted 'unknown' only — RED."""
from neurobrix.kernels import autotune_certified as C


def test_the_label_reads_the_version_and_a_non_cuda_backend():
    assert C.proof_backend({"backend": {"triton": "3.6.0", "name": "cuda"}}) == "triton 3.6.0"
    assert C.proof_backend({"backend": {"triton": "3.7.0", "name": "metal"}}) == "triton 3.7.0 metal"
    assert C.proof_backend({}) is None and C.proof_backend(None) is None


def test_the_label_carries_the_out_of_tree_backend_hash():
    """An out-of-tree backend (mps under triton-ext) records its own source
    hash; the label must fold it in so two triton-ext builds that share the
    distribution version are told apart. In-tree (cuda/amd, no hash) is
    unchanged. RED: label built from triton+name only → both builds share it."""
    ident = {"triton": "3.8.0+gitPIN", "name": "mps",
             "backend_hash": "msl-v0.1-aabbccddeeff0011"}
    lab = C.proof_backend({"backend": ident})
    assert lab == "triton 3.8.0+gitPIN mps msl-v0.1-aab", lab
    # cuda / no hash: unchanged from before the fix (the rack's proofs untouched)
    assert C.proof_backend({"backend": {"triton": "3.8.0+gitPIN", "name": "cuda"}}) \
        == "triton 3.8.0+gitPIN"


def test_two_backend_builds_split_on_the_hash_though_the_version_is_one():
    """The whole point: same triton distribution version, different out-of-tree
    backend build → different generator labels, so the gate re-certifies."""
    a = C.proof_backend({"backend": {"triton": "3.8.0+gitPIN", "name": "mps",
                                     "backend_hash": "msl-v0.1-5439436aaaaa"}})
    b = C.proof_backend({"backend": {"triton": "3.8.0+gitPIN", "name": "mps",
                                     "backend_hash": "msl-v0.1-b9d5c06dbbbb"}})
    assert a != b, (a, b)


def test_the_identity_raises_rather_than_default_to_cuda(monkeypatch):
    """A target that cannot name its backend is an ERROR, not 'cuda' (defaulting
    to cuda once refused all 945 Apple entries). generator_identity raises, and
    running_generator turns that into 'cannot say' (None) — never a wrong label."""
    import types
    from neurobrix.kernels import launcher as L

    monkeypatch.setattr(L, "target", lambda: types.SimpleNamespace(backend=None),
                        raising=True)
    import pytest
    with pytest.raises(RuntimeError, match="not 'cuda'"):
        C.generator_identity()
    # the gate stays alive and answers "cannot say", so it refuses nothing
    assert C.running_generator() is None


def test_the_live_identity_carries_a_hash_iff_the_backend_is_out_of_tree():
    """On this machine: if the active Triton backend is out of tree (mps under
    triton-ext), the identity carries a non-empty backend_hash; if in-tree
    (cuda/amd) it carries none. Either way the label round-trips."""
    import pytest
    try:
        ident = C.generator_identity()
    except RuntimeError:
        pytest.skip("no Triton target on this machine to identify")
    name = ident.get("name")
    if name in ("cuda", "hip"):
        assert "backend_hash" not in ident
    else:
        assert ident.get("backend_hash"), ident
        assert str(ident["backend_hash"])[:12] in C.running_generator()


def test_an_entry_collects_the_generators_of_its_primary_and_variant_proofs():
    entry = {"proof": {"backend": {"triton": "3.6.0", "name": "cuda"}},
             "variants": {"32": {"proof": {"backend": {"triton": "3.8.1", "name": "cuda"}}}}}
    assert C.proof_backends(entry) == {"triton 3.6.0", "triton 3.8.1"}


def test_the_live_directory_s_proofs_are_all_labelled():
    """Every proof the trunk carries names its generator (no 'unknown')."""
    import json
    labels = {}
    for p in C.files():
        for entry in (json.loads(p.read_text(encoding="utf-8")).get("entries") or {}).values():
            for lab in (C.proof_backends(entry) or {"unknown"}):
                labels[lab] = labels.get(lab, 0) + 1
    assert labels and "unknown" not in labels, labels


def test_a_proof_under_another_generator_does_not_cover_when_the_running_one_is_asked():
    """`--reprove-generator`: an entry proven under triton 3.6.0 is not coverage
    for a certifier running triton 3.8.0; the same entry covers when the label
    matches or when no generator is asked. Injection: `need_generator` ignored
    in `entry_covers` → the first assertion held True — RED."""
    entries = {"k": {"config": {}, "proof": {"backend": {"triton": "3.6.0", "name": "cuda"},
                                              "machine": {"device": {"memory_mb": 32768}}}}}
    assert C.entry_covers(entries, "k", 32, need_generator="triton 3.8.0") is False
    assert C.entry_covers(entries, "k", 32, need_generator="triton 3.6.0") is True
    assert C.entry_covers(entries, "k", 32) is True
