"""The bucket ladder (kernels/autotune_bucket.py): the values the measurement chose. What this
test would do if the code were wrong: place 57 in a 16-wide bucket (the batched GEMM lost up
to 16.7 % there), or bucket a profile that declares no ladder."""
from neurobrix.kernels.autotune_bucket import bucket, bucket_of, ladder_for, parse_ladder

LADDER = parse_ladder([{"up_to": 64, "step": 1}, {"up_to": 256, "step": 16},
                       {"up_to": 1024, "step": 32}, {"up_to": 8192, "step": 128}, {"up_to": None, "step": 512}])


def test_exact_under_64_then_the_measured_steps():
    assert [bucket(v, LADDER) for v in (1, 19, 57, 64)] == [1, 19, 57, 64]
    assert [bucket(v, LADDER) for v in (65, 80, 129, 256)] == [80, 80, 144, 256]
    assert [bucket(v, LADDER) for v in (257, 1000, 1024)] == [288, 1024, 1024]
    assert [bucket(v, LADDER) for v in (1025, 4097, 8192)] == [1152, 4224, 8192]
    assert bucket(8193, LADDER) == 8704 and bucket(32760, LADDER) == 32768


def test_a_profile_without_a_ladder_keeps_the_key_exact():
    assert ladder_for("M", {}) == [(None, 1)]
    assert bucket_of("M", 57, {}) == 57 and bucket_of("M", 4097, {}) == 4097


def test_the_volta_profile_declares_the_measured_ladder():
    import yaml
    from pathlib import Path
    import neurobrix
    p = Path(neurobrix.__file__).parent / "config" / "vendors" / "nvidia" / "volta.yml"
    prof = yaml.safe_load(p.read_text())
    assert bucket_of("M", 57, prof) == 57 and bucket_of("M", 65, prof) == 80 and bucket_of("N", 1025, prof) == 1152
