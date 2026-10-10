"""A synthetic DAG written the way a real graph.json is.

Every op of a real graph carries the tensors it receives in its `attributes` args (81 514 ops over
40 cached components, 2026-10-10: none without), and the one liveness rule
(`core/runtime/liveness.py`) reads THOSE, not `input_tensor_ids`. A fixture that names only
`input_tensor_ids` describes ops that read nothing: every activation dies at its producer and each
estimate below prices a graph that does not exist.
"""


def as_graph(dag: dict) -> dict:
    """Write each op's `input_tensor_ids` into its `attributes` args, in order; returns `dag`.

    Rewritten at every call, so a fixture edited after it is built is passed through again."""
    for op in dag["ops"].values():
        attrs = op.setdefault("attributes", {})
        attrs["args"] = [{"type": "tensor", "tensor_id": t} for t in op.get("input_tensor_ids", [])]
    return dag
