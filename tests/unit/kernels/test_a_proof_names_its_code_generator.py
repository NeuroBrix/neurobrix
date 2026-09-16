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
