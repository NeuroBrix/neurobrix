"""NBX_DISABLE_MOE_FUSION on a graph with a routed MoE is refused by name, not left to die in a split of the build-time
routing (granite on the Mac 23/11: "split_with_sizes expects split_sizes to sum exactly to 648", PyTorch branch;
"Cannot broadcast (184, 1024) and (648, 1)", Triton). A graph without a routed MoE is returned untouched."""
import pytest

from neurobrix.core.runtime.graph import moe_fusion
from neurobrix.pytorch.neurotax import moe_tokens


def test_a_routed_graph_is_refused_and_a_plain_one_passes(monkeypatch):
    monkeypatch.setenv("NBX_DISABLE_MOE_FUSION", "1")
    monkeypatch.setattr(moe_fusion, "_wide_topks", lambda ops, order: ["aten.topk::0"])
    with pytest.raises(RuntimeError, match="carries the build-time routing as constants"):
        moe_fusion._unfused({"ops": {}, "execution_order": []}, "llm", False)
    monkeypatch.setattr(moe_fusion, "_wide_topks", lambda ops, order: [])
    dag = {"ops": {}, "execution_order": []}
    assert moe_fusion._unfused(dag, "llm", False) is dag
    assert moe_fusion._unfused(dag, "image", False) is dag


def test_the_switch_reaches_the_refusal_through_the_fusion_pass_itself(monkeypatch):
    """qa 1298-1: through detect_and_fuse_moe, the door every caller uses: a switch that returned the graph silently
    there (the mutant) would pass the test of _unfused alone; here it is red."""
    monkeypatch.setenv("NBX_DISABLE_MOE_FUSION", "1")
    monkeypatch.setattr(moe_fusion, "_wide_topks", lambda ops, order: ["aten.topk::0"])
    with pytest.raises(RuntimeError, match="carries the build-time routing as constants"):
        moe_fusion.detect_and_fuse_moe({"ops": {}, "execution_order": []}, "llm", tokens=moe_tokens())
    with pytest.raises(RuntimeError, match="carries the build-time routing as constants"):
        moe_fusion.detect_and_fuse_moe({"ops": {}, "execution_order": []}, "image", declared=True, tokens=moe_tokens())
    monkeypatch.setattr(moe_fusion, "_wide_topks", lambda ops, order: [])
    dag = {"ops": {}, "execution_order": []}
    assert moe_fusion.detect_and_fuse_moe(dag, "llm", tokens=moe_tokens()) is dag
