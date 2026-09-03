"""Pin for the session graph-source rule (serve TTFT reconciliation,
2026-09-03): a session reads its input contract from the executor's
parsed DAG when the executor exists — the graph.json file is opened only
on the cold path (no executor yet). Both engines carry the same rule as
separate mirrors (R30); each is pinned here.

Why it matters: the 30B MoE graph.json is 244 MB (4 s to parse); re-read
per request it was 8.3 s of the 44.7 s serve TTFT at 8.3k context.
"""
import json
from pathlib import Path

import pytest

from neurobrix.core.flow.autoregressive import _session_dag as compiled_session_dag
from neurobrix.triton.flow.autoregressive import _session_dag as triton_session_dag


class _Exec:
    def __init__(self, dag):
        self.dag = dag


@pytest.mark.parametrize("session_dag", [compiled_session_dag, triton_session_dag],
                         ids=["compiled", "triton"])
def test_live_executor_dag_wins_and_file_is_never_opened(tmp_path, session_dag):
    dag = {"input_tensor_ids": ["input::input_ids"], "tensors": {}, "ops": {}}
    missing = tmp_path / "components" / "model" / "graph.json"   # does not exist
    assert session_dag(_Exec(dag), missing, "model") is dag


@pytest.mark.parametrize("session_dag", [compiled_session_dag, triton_session_dag],
                         ids=["compiled", "triton"])
def test_cold_path_reads_the_file(tmp_path, session_dag):
    p = tmp_path / "graph.json"
    p.write_text(json.dumps({"input_tensor_ids": ["input::x"], "tensors": {}, "ops": {}}))
    assert session_dag(None, p, "model")["input_tensor_ids"] == ["input::x"]


@pytest.mark.parametrize("session_dag", [compiled_session_dag, triton_session_dag],
                         ids=["compiled", "triton"])
def test_live_executor_without_dag_is_zero_fallback(tmp_path, session_dag):
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        session_dag(_Exec(None), tmp_path / "graph.json", "model")


@pytest.mark.parametrize("session_dag", [compiled_session_dag, triton_session_dag],
                         ids=["compiled", "triton"])
def test_cold_path_missing_file_is_zero_fallback(tmp_path, session_dag):
    with pytest.raises(RuntimeError, match="graph.json not found"):
        session_dag(None, tmp_path / "graph.json", "model")


def test_flow_registry_still_binds_the_handler_classes():
    """The helpers live ABOVE the @register_flow decorators: a def placed
    between a decorator and its class silently re-binds the registry key to
    the helper (caught by the doctrine review, 2026-09-03)."""
    from neurobrix.core.flow import FLOW_REGISTRY
    from neurobrix.core.flow.autoregressive import AutoregressiveHandler
    assert FLOW_REGISTRY["autoregressive_generation"] is AutoregressiveHandler
