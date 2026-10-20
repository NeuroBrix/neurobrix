"""State engine step 6 (docs/design/state-engine.md §3.3 point 6, §4.8 step 6): Prism's profile is NODES (machines: their
devices, their host memory) and LINKS between them (bandwidth, latency); a single machine is the one-node case, never a
special one. No cluster profile exists yet, so every plan of today is the one-node plan.

What this test does if the code is wrong: the one-node cell fails when a profile without `nodes` gets none, or a node
that misses a device or the profile's CPU; the cluster cell fails when declared nodes and links are not read, or a
device is looked up on the wrong node; the refusal cells fail when a device on no node, a device on two nodes, or a
link to an undeclared node passes silently.
"""
import pytest
import yaml

from neurobrix.core.prism import loader as L
from neurobrix.core.prism.structure import ClusterNode, DeviceSpec, NodeLink, PrismProfile


def _dev(i):
    return DeviceSpec(index=i, name="gpu", memory_mb=16384, compute_capability="0.0", supports_dtypes=["float32"])


def test_a_profile_without_nodes_is_one_node_with_every_device_and_its_cpu():
    p = PrismProfile(id="one", vendor="v", devices=[_dev(0), _dev(1)], cpu="the-cpu")
    assert len(p.nodes) == 1 and p.nodes[0].device_indices == [0, 1] and p.nodes[0].cpu == "the-cpu"
    assert p.node_of(1) is p.nodes[0] and p.node_links == []


def test_a_cluster_profile_is_read_as_its_nodes_and_links(tmp_path, monkeypatch):
    doc = {"id": "two-macs", "vendor": "apple",
           "devices": [{"index": 0, "memory_mb": 24576}, {"index": 1, "memory_mb": 24576}],
           "nodes": [{"id": "mac-a", "devices": [0]}, {"id": "mac-b", "devices": [1]}],
           "node_links": [{"between": ["mac-a", "mac-b"], "bandwidth_gbps": 80, "latency_us": 5, "tech": "rdma-tb5"}]}
    (tmp_path / "two-macs.yml").write_text(yaml.safe_dump(doc))
    monkeypatch.setattr(L, "HARDWARE_DIR", tmp_path)
    p = L.load_profile("two-macs")
    assert [n.id for n in p.nodes] == ["mac-a", "mac-b"] and p.node_of(1).id == "mac-b"
    assert p.node_links == [NodeLink("mac-a", "mac-b", 80.0, 5.0, "rdma-tb5")]


@pytest.mark.parametrize("nodes, links, words", [
    ([ClusterNode("a", [0])], [], "unplaced \\[1\\]"),
    ([ClusterNode("a", [0, 1]), ClusterNode("b", [1])], [], "in two nodes \\[1\\]"),
    ([ClusterNode("a", [0, 1])], [NodeLink("a", "z", 1.0, 1.0)], "names a node"),
])
def test_a_device_on_no_node_on_two_nodes_or_a_link_to_nowhere_is_refused_by_name(nodes, links, words):
    with pytest.raises(ValueError, match=words):
        PrismProfile(id="bad", vendor="v", devices=[_dev(0), _dev(1)], nodes=nodes, node_links=links)
