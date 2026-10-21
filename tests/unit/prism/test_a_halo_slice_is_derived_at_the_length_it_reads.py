"""A halo stretch's slice runs at the rows it READS (its own plus the halo on each side, clipped at the axis' ends):
the executor and the census take those lengths from one function, so a slice's keys are derived at the length the
runtime forms them — never at the slice's own length, which no launch of a halo stretch ever has."""
import sys
from pathlib import Path

from neurobrix.core.prism.chunked_region import chunk_extents, slice_reads

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))


def test_a_slice_reads_its_rows_widened_by_the_halo_and_clipped_at_the_ends():
    assert slice_reads(10, 3, 2) == [(0, 4, 0, 6), (4, 4, 2, 8), (8, 2, 6, 4)]


def test_without_a_halo_a_slice_reads_its_own_rows():
    assert [(s, n) for s, n, _a, _b in slice_reads(10, 3, 0)] == chunk_extents(10, 3)
    assert all((s, n) == (a, b) for s, n, a, b in slice_reads(10, 3, 0))


def test_the_executor_and_the_census_read_the_same_function():
    import inspect
    from neurobrix.core.strategies import chunked_piece
    import derived_census
    assert "slice_reads(" in inspect.getsource(chunked_piece.ChunkedPiece._run_pass)   # the pass loop
    assert "slice_reads(" in inspect.getsource(derived_census.sliced_bindings)
