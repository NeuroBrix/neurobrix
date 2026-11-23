"""The matrix tile lives ONCE, in the library's half of each device (`block_sizes.bmm`, nbxtensor/backend/devices):
the engine's vendor files carried a `block_sizes.gemm` of their own that no kernel read and that disagreed with the
library on ampere and hopper (hopper: engine 128/256/64, library 64/128/32) — a dead duplicate that disagrees is the
fault of two copies (supervisor 23/11 01:01, OWNER RESET 22/11 point 6: one key, one place).

Red on main 9eb471d: 41 engine vendor files state the tile and `core.config.get_gemm_config` reads it."""
from pathlib import Path

import yaml

import neurobrix.core.config as config

VENDORS = Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "config" / "vendors"


def test_no_engine_vendor_file_states_a_matrix_tile():
    stating = [str(f.relative_to(VENDORS)) for f in sorted(VENDORS.glob("*/*.yml"))
               if "gemm" in ((yaml.safe_load(f.read_text()) or {}).get("block_sizes") or {})]
    assert not stating, stating


def test_the_engine_has_no_reader_of_its_own_tile():
    assert not hasattr(config, "get_gemm_config")
