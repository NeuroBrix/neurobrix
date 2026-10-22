"""A component the flow feeds a stack of independent items runs them as its vendor's reference does — in passes of the
declared size — priced per pass and joined on the batch axis.

VACE's control pair (inactive, reactive) is built by the CLI on the encoder's batch axis and was encoded in ONE pass of
two; diffusers' WanVACEPipeline.prepare_video_latents encodes them as two calls of one item. The stacked pass doubles
the encoder's live set — on a 24 GB Mac the chunked encoder ran out of memory where the vendor's way fits (2026-10-18).
The container declares the vendor's passes (`vace_control_conditioning.items_per_pass`, from the registry with its
source); the bindings price the component at that batch (`FlowBindings.pass_batch`), the plan records the split
(`ExecutionPlan.batch_split_components`), and the executor runs the items group by group (`Executor._run_in_item_passes`).

What each test would do if the code were wrong: a declaration ignored prices the pair (red at pass_batch); a plan that
does not record the split has an empty field (red); an executor that feeds the whole stack to one pass, or joins the
outputs in another order, returns the wrong rows (red); a declared size that does not divide the stack, taken anyway,
passes the refusal test's `raises` no more (red).
"""
from __future__ import annotations

import dataclasses
import types

import pytest
import torch

from neurobrix.core.prism.flow_bindings import FlowBindings

PAIR = {"condition_component": "vae_encoder", "vace_layers": 15, "mask_channels": 64, "z_dim": 16}


def _topology(items_per_pass=None):
    flag = dict(PAIR)
    if items_per_pass is not None:
        flag["items_per_pass"] = items_per_pass
    return {"flow": {"loop": {"components": ["transformer"], "state_variable": "transformer.hidden_states"}},
            "connections": [{"from": "global.image", "to": "vae_encoder.x"}, {"from": "vae_encoder.output_0", "to": "transformer.control"}],
            "extracted_values": {"transformer": {"vace_control_conditioning": flag}, "vae_encoder": {}, "vae": {}}}


def _bindings(tmp_path, items_per_pass=None, name="a-test-container"):
    return FlowBindings(_topology(items_per_pass), tmp_path, name)


def test_the_declared_passes_set_the_batch_the_component_is_priced_at(tmp_path):
    fb = _bindings(tmp_path, items_per_pass=1, name="a-test-container-split")
    assert fb.vace_control_batch("vae_encoder") == 2 and fb.pass_batch("vae_encoder") == 1
    assert fb.split_batches(["vae_encoder", "transformer", "vae"]) == {"vae_encoder": {"fed": 2, "per_pass": 1}}
    dag = {"component_name": "vae_encoder", "symbolic_context": {"symbols": {"s0": {"name": "batch"}, "s1": {"name": "time"}}}}
    ic = types.SimpleNamespace(batch_size=2, guidance_passes=2, flow=fb)
    assert fb.overrides(dag, ic).get("s0") == 1, "the pair's batch symbol must be the pass's"


def test_without_a_declaration_the_stack_runs_in_one_pass_as_before(tmp_path):
    fb = _bindings(tmp_path, name="a-test-container-whole")
    assert fb.pass_batch("vae_encoder") == 2 and fb.split_batches(["vae_encoder"]) == {}
    assert fb.pass_batch("vae") is None


def test_a_pass_size_that_does_not_divide_the_stack_is_refused_by_name(tmp_path):
    fb = _bindings(tmp_path, items_per_pass=3, name="a-test-container-three")
    with pytest.raises(ValueError, match="does not divide"):
        fb.pass_batch("vae_encoder")


class _Fake:
    """An executor whose cascade records what each pass received and returns twice its x."""
    def __init__(self):
        self.calls = []

    def _run_component_once(self, comp_name, phase, executor, inputs):
        self.calls.append({k: (v.clone() if hasattr(v, "clone") else v) for k, v in inputs.items()})
        return {"out": inputs["x"] * 2, "count": inputs["count"]}


def test_the_executor_runs_the_items_group_by_group_and_joins_them_in_order():
    from neurobrix.core.runtime.executor import RuntimeExecutor as Executor
    fake = _Fake()
    x = torch.arange(4 * 3, dtype=torch.float32).view(4, 3)
    mask = torch.ones(1, 3)
    out = Executor._run_in_item_passes(fake, "enc", "pre_loop", None, {"x": x, "mask": mask, "count": 7}, fed=4, per_pass=2)
    assert len(fake.calls) == 2
    assert torch.equal(fake.calls[0]["x"], x[0:2]) and torch.equal(fake.calls[1]["x"], x[2:4])
    assert all(torch.equal(c["mask"], mask) and c["count"] == 7 for c in fake.calls), "an input the stack does not batch is handed whole"
    assert torch.equal(out["out"], x * 2) and out["count"] == 7
    with pytest.raises(ValueError, match="do not cover"):
        Executor._run_in_item_passes(fake, "enc", "pre_loop", None, {"x": x}, fed=4, per_pass=3)


def _cuda() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _cuda(), reason="the Triton kind lives on a card")
def test_the_join_is_the_run_s_own_tensor_kind():
    from neurobrix.core.runtime.executor import RuntimeExecutor as Executor
    from neurobrix.kernels.nbx_tensor import NBXTensor
    fake = _Fake()
    x = NBXTensor.from_numpy(torch.arange(2 * 3, dtype=torch.float32).view(2, 3).numpy()).to_cuda()
    out = Executor._run_in_item_passes(fake, "enc", "pre_loop", None, {"x": x, "count": 1}, fed=2, per_pass=1)
    assert isinstance(out["out"], NBXTensor)
    assert torch.equal(torch.from_numpy(out["out"].numpy()), torch.arange(6, dtype=torch.float32).view(2, 3) * 2)


def test_the_plan_records_the_split_and_prices_the_encoder_per_pass(monkeypatch):
    """VACE on the Mac's pinned profile: with the declaration the encoder's chunk price is one item's and the plan says
    so; the same request without it prices the pair."""
    from neurobrix.core.prism import PrismSolver
    from neurobrix.core.prism.solver import plan_record, explain_plan
    from neurobrix.nbx.container import NBXContainer
    from tests.unit.prism._pinned_machine import APPLE_M4_PRO, container_root, impose_rung, pin_host, profile
    import json
    root = container_root("Wan2.1-VACE-1.3B")
    if not (root / "topology.json").exists():
        pytest.skip("Wan2.1-VACE-1.3B is not in this machine's cache")
    pin_host(monkeypatch, 24576, 11198, "the Mac's reading"); impose_rung(monkeypatch, 18432)
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    import sys
    sys.path.insert(0, "tools")
    from trace_request import derived_request
    c = NBXContainer.load(str(root)); man = c.get_manifest() or {}
    args = create_parser().parse_args(["run", "--model", "Wan2.1-VACE-1.3B", *derived_request("Wan2.1-VACE-1.3B"), "--compiled"])
    ic = request_input_config(args, man, man.get("family"), c.cache_path)
    topo = json.loads((root / "topology.json").read_text())
    # both cases stated here: the container itself declares items_per_pass since its single write (2026-10-18),
    # so the request as cached is already the split one — the whole case removes the declaration
    vcc = topo["extracted_values"]["transformer"]["vace_control_conditioning"]
    whole_topo = json.loads(json.dumps(topo))
    whole_topo["extracted_values"]["transformer"]["vace_control_conditioning"].pop("items_per_pass", None)
    vcc["items_per_pass"] = 1
    split_ic = dataclasses.replace(ic, flow=FlowBindings(topo, root, "Wan2.1-VACE-1.3B-split-test"))
    whole_ic = dataclasses.replace(ic, flow=FlowBindings(whole_topo, root, "Wan2.1-VACE-1.3B-whole-test"))
    def solve(input_config):
        # the encoder's priced bytes, as the solver computes them (`_compute_memory`)
        s, priced = PrismSolver(), {}
        compute = s._compute_memory
        def spy(*a, **k):
            out = compute(*a, **k)
            if isinstance(out, dict) and "vae_encoder" in out:
                priced["vae_encoder"] = out["vae_encoder"].total_bytes
            return out
        s._compute_memory = spy
        return s.solve_smart(c, profile(APPLE_M4_PRO), input_config, mode="compiled"), priced["vae_encoder"]
    whole, whole_bytes = solve(whole_ic)
    split, split_bytes = solve(split_ic)
    assert whole.batch_split_components == {}
    assert split.batch_split_components == {"vae_encoder": {"fed": 2, "per_pass": 1}}
    assert "batch_split_components" in plan_record(split) and any("passes" in l and "vae_encoder" in l for l in explain_plan(split).split("\n"))
    # One item per pass prices under the stacked pair. Re-stated 2026-10-22 on the measurement: the cell had pinned the
    # encoder streamed in chunks because it was priced at the guidance-doubled batch (33 300.7 MB); the encoder is not
    # inside the guided loop, and Metal MEASURED it whole at this request in passes of one item, its pool at 15 762 MB
    # (Mac 04:46; CUDA 24 395 MB at its gate's request, Dell 04:15) — priced 16 704.0 MB once the guidance batch is
    # undone outside the loop. So the split's price is judged, not a streaming schedule that no longer exists.
    assert split_bytes < whole_bytes, (split_bytes, whole_bytes)
