"""The container's graphs are at the re-propagation fixed point, or the engine refuses it.

The compiled sequence carried a compensation for graphs whose trace left a cross-branch
dimension literal: it re-keyed an expand / view / reshape literal equal to the trace value of
the only expression of that value in the graph. On the graphs the build toolchain's single
write leaves at its symbolic fixed point it could only corrupt literals, and it did
(2026-10-04): Ming-Lite-Omni-1.5's GQA heads 4 / 16 became 2176 at a 275-token prompt once
the MoE token counts were the only expressions of trace 3 and 4, and Sana_1600M_4Kpx_BF16's
VAE `view::31` merged its 4096 channels into an expression of trace 1024. The pass is gone;
the defect it compensated is fixed where it is born, in the trace.

The writer records that fact on every component it leaves at the fixed point:
`extracted_values[<component>]["symbolic_fixed_point"]` in topology.json, its value the
writer's revision (an existing free-form per-component dict, no new field — R18). A
container carrying a graph without it was not written at the fixed point and may still
depend on the deleted compensation, so it is refused by name before any weight I/O, like a
container of another NeuroTax version (`nbx.neurotax.NEUROTAX_VERSION`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

#: The per-component declaration the single write carries in `extracted_values`.
FIXED_POINT_FLAG = "symbolic_fixed_point"


def unstamped_components(cache_path, topology: Dict[str, Any]) -> List[str]:
    """Components of the container at ``cache_path`` that carry a graph and no fixed-point
    declaration, sorted."""
    values = topology.get("extracted_values")
    values = values if isinstance(values, dict) else {}
    missing = []
    for graph in sorted((Path(cache_path) / "components").glob("*/graph.json")):
        name = graph.parent.name
        declared = values.get(name)
        if not (isinstance(declared, dict) and declared.get(FIXED_POINT_FLAG)):
            missing.append(name)
    return missing


def refuse_unstamped(cache_path, topology: Dict[str, Any]) -> None:
    """Raise, naming the container and each component, unless every graph declares the fixed
    point."""
    missing = unstamped_components(cache_path, topology)
    if missing:
        raise RuntimeError(
            f"SYMBOLIC FIXED POINT: this container's graph(s) {missing} carry no "
            f"'{FIXED_POINT_FLAG}' declaration in topology.json extracted_values, so they were "
            f"not written at the symbolic fixed point this engine runs.\n"
            f"  Container: {cache_path}\n"
            f"  FIX: install the container published for this engine "
            f"(`neurobrix remove <model> && neurobrix import <org>/<model>`).")
