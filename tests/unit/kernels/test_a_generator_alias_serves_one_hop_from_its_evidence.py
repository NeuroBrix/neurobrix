"""A generator alias (`generator_aliases.json` in a profile's certified directory) lets a build proven to produce the SAME
compiled code be served the certifications of another (the owner, 2026-11-08 21:07: the macOS 14 Metal wheels, whose
235-236 kernels compiled byte-identical .ttir/.ttgir/.msl/.metallib to the published macOS 15 build's, are not
re-proven key by key). Decided between the Dell and the Mac 21:11: the file is scoped to its directory; a proof naming
the running generator always wins; otherwise ONE hop to the alias's `to`, never a chain; the serving says it; the gate
refuses a malformed file.

What this test does if the code is wrong: red when an aliased generator is served nothing (no alias read), when the
alias wins over a proof that names the running generator, when a chain is followed (two hops), when the alias file is
read as a certified file by `files()`, or when the gate accepts a file without evidence, with a chain, with a duplicate
`from`, or whose `to` no proof in the directory names. No card: the running generator is stated."""
import json

import pytest

from nbxtensor.ops import autotune_certified as C

PUB = {"triton": "3.8.0+git4a15f415.nbx1", "name": "mps", "backend_hash": "content-0c61dc742284af40"}
MAC14 = {"triton": "3.8.0+git4a15f415.nbx1", "name": "mps", "backend_hash": "content-3a38a05dfda2da61"}
OTHER = {"triton": "3.8.0+git4a15f415.nbx1", "name": "mps", "backend_hash": "content-ffffffffffffffff"}
EVIDENCE = {"wheels": "macosx_14_0 cp310-cp313", "sets": 4, "kernels": 235,
            "path": "campagnes/2026_11_08_metal_wheels_macos14/proof/alias_evidence.tsv", "sha256": "ab" * 32}


def _label(ident):
    return C.proof_backend({"backend": ident})


def _alias(frm=MAC14, to=PUB, **over):
    entry = {"from": frm, "to": to, "evidence": EVIDENCE, "date": "2026-11-08",
             "decided_by": "owner 2026-11-08 21:07 via the supervisor"}
    entry.update(over)
    return entry


def _dir(tmp_path, aliases, proof_gens=(PUB,)):
    d = tmp_path / "apple" / "apple_m4_pro"
    d.mkdir(parents=True)
    cert = {"config": {"BLOCK": 64}, "proof": {"backend": proof_gens[0], "memory_gb": 18},
            "also_proven": [{"backend": g, "memory_gb": 18} for g in proof_gens[1:]]}
    (d / "k.fp32.json").write_text(json.dumps({"entries": {"(1,)": cert}}))
    (d / C.GENERATOR_ALIASES).write_text(json.dumps({"aliases": aliases}))
    return d, cert


def test_an_aliased_generator_is_served_its_target_s_certification(tmp_path):
    d, cert = _dir(tmp_path, [_alias()])
    got = C.through_generator_alias(cert, _label(MAC14), d)
    assert got is not None and got[0]["config"] == {"BLOCK": 64} and got[1] == (_label(MAC14), _label(PUB))


def test_a_proof_naming_the_running_generator_wins_over_the_alias(tmp_path):
    d, cert = _dir(tmp_path, [_alias()], proof_gens=(PUB, MAC14))
    assert C.certification_for_generator(cert, _label(MAC14)) is not None     # served directly, the alias never asked


def test_one_hop_only(tmp_path):
    d, cert = _dir(tmp_path, [_alias(OTHER, MAC14), _alias(MAC14, PUB)])
    assert C.through_generator_alias(cert, _label(OTHER), d) is None          # OTHER -> MAC14 has no proof: no 2nd hop


def test_the_alias_file_is_not_a_certified_file(tmp_path):
    d, _ = _dir(tmp_path, [_alias()])
    assert [p.name for p in C.files(tmp_path)] == ["k.fp32.json"]


@pytest.mark.parametrize("bad, said", [
    ([_alias(evidence={})], "evidence"),
    ([_alias(OTHER, MAC14), _alias(MAC14, PUB)], "chain"),
    ([_alias(), _alias()], "duplicate"),
    ([_alias(MAC14, OTHER)], "no proof"),
    ([_alias(decided_by="")], "decided_by"),
])
def test_the_gate_refuses_a_malformed_alias_file(tmp_path, bad, said):
    d, _ = _dir(tmp_path, bad)
    problems = C.validate_generator_aliases(json.loads((d / C.GENERATOR_ALIASES).read_text()), d)
    assert any(said in p for p in problems), problems


def test_the_gate_accepts_a_well_formed_alias_file(tmp_path):
    d, _ = _dir(tmp_path, [_alias()])
    assert C.validate_generator_aliases(json.loads((d / C.GENERATOR_ALIASES).read_text()), d) == []


def test_the_alias_line_is_a_debug_line_not_the_users_terminal(capsys, caplog):
    import logging
    with caplog.at_level(logging.DEBUG, logger=C.__name__):
        C.say_aliased("k", _label(MAC14), _label(PUB))
    assert capsys.readouterr().out == ""                              # nothing on a plain run's terminal
    assert any("served through the generator alias" in r.getMessage() for r in caplog.records)
