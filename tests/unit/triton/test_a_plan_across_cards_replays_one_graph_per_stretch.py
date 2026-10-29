"""A recorded plan that crosses cards (an expert store's card-held tail) replays one CUDA graph per stretch of actions on
one card, the card switches and the recorded syncs run directly between them, in the recorded order (S3, 2026-10-28:
Qwen3-30B-A3B on 4 V100, 1 845 launches as 121 graphs, 184 -> 175 ms a token, text identical). No card."""
from nbxtensor.backend import replay_graph as RG


class _Plan:
    made = []

    def __init__(self, records, dev):
        self.records, self.dev, self.launches = list(records), dev, sum(1 for t, _ in records if t == "k")
        _Plan.made.append(self)

    def launch(self):
        pass

    def destroy(self):
        pass


def test_the_stretches_follow_the_card_switches_and_the_syncs(monkeypatch):
    _Plan.made = []
    monkeypatch.setattr(RG, "CapturedPlan", _Plan)
    recs = [("k", "a0"), ("k", "a1"), ("d", (1,)), ("m", "x"), ("k", "b0"), ("y", (1,)), ("d", (2,)), ("k", "c0"),
            ("d", (2,)), ("k", "c1"), ("d", (0,)), ("k", "a2")]
    p = RG.SegmentedPlan(recs, 0)
    assert [(q.dev, [r for _t, r in q.records]) for q in _Plan.made] == [
        (0, ["a0", "a1"]), (1, ["x", "b0"]), (2, ["c0", "c1"]), (0, ["a2"])], [(q.dev, q.records) for q in _Plan.made]
    assert [t for t, _ in p.steps] == ["g", "d", "g", "y", "d", "g", "d", "g"], p.steps
    assert p.launches == 6 and p.stretches == 4


def test_an_action_with_no_graph_form_is_refused(monkeypatch):
    monkeypatch.setattr(RG, "CapturedPlan", _Plan)
    try:
        RG.SegmentedPlan([("k", "a"), ("?", ())], 0)
    except RG.GraphReplayError as e:
        assert "no graph form" in str(e)
    else:
        raise AssertionError("an unknown action was captured")
