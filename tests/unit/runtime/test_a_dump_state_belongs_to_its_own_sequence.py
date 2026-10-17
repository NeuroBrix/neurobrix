"""The compiled sequence's per-op dump state (NBX_DUMP_TIDS, a diagnostic) belongs to ONE sequence.

Keyed by id(self) alone, an entry outlived its sequence and the next sequence allocated at that address inherited its
pass counter and its seen set: its first pass was numbered after the dead one's and its ops already "seen", so the
dump skipped them (found 2026-10-16 during the Wan2.2 investigation). What the test would do if the code were wrong:
the stale entry planted at the new sequence's address is returned as its own — red.
"""
from neurobrix.core.runtime.graph import compiled_sequence as CS


class _Seq:
    __slots__ = ("_tensor_id_to_slot",)

    def __init__(self):
        self._tensor_id_to_slot = {"t0": 0}


def test_a_stale_entry_at_the_same_address_is_not_inherited():
    seq = _Seq()
    CS._DUMP_STATE_NATIVE[id(seq)] = {"owner": object(), "seen": {"aten.add::0"}, "records": [1], "pass": 7,
                                      "slot_to_tid": {}}
    st = CS._dump_state_native(seq)
    assert st["owner"] is seq and st["seen"] == set() and "pass" not in st and st["slot_to_tid"] == {0: "t0"}
    st["pass"] = 0
    assert CS._dump_state_native(seq) is st                     # its own entry, kept across its runs
    CS._DUMP_STATE_NATIVE.pop(id(seq), None)
