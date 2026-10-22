"""`certify --models` reads only the census rows of the named containers: a kernel change re-sweeps every key of a
kernel (~7 500 Apple conv keys at ~29 s), and the keys of the models that need it first are a subset of the table."""
import json

from neurobrix.kernels import autotune_certify as A
from neurobrix.kernels import census_table as T


def test_the_filter_keeps_exactly_the_named_models_keys():
    path = T.table_path("apple", "apple_m4_pro", 18)
    rows = [r for r in T.read(path) if r.get("model") == "CogVideoX-2b"]
    want = {(r["kernel"], r["key"]) for r in rows}
    got = A.census("apple", "apple_m4_pro", 18, ["CogVideoX-2b"])
    got_pairs = {(q, repr(k)) for q, ks in got.items() for k in ks}
    assert rows and len(got_pairs) == len(want)
    assert sum(len(v) for v in got.values()) < sum(len(v) for v in A.census("apple", "apple_m4_pro", 18).values())


def test_no_filter_reads_the_whole_table():
    whole = A.census("apple", "apple_m4_pro", 18)
    assert A.census("apple", "apple_m4_pro", 18, None) == whole
