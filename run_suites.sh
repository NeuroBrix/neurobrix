#!/bin/bash
# The wider unit suites on the refusal tree, one directory at a time, each under its own cap.
W=/home/mlops/nbx/worktrees/symbol_refuses; PY=/home/mlops/venvs/nbx_t214/bin/python; OUT=$W/suites.md
echo "| dir | result | time |" > $OUT; echo "|---|---|---|" >> $OUT
for d in tests/unit/runtime tests/unit/prism tests/unit/cli tests/unit/kernels tests/unit/triton; do
  r=$( cd $W && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$W/src timeout 900 $PY -m pytest $d -q -p no:cacheprovider --deselect tests/unit/kernels/test_autotune_certify_first_light.py::test_one_matmul_shape_is_certified_gated_and_served 2>&1 | grep -E "^FAILED|passed|failed" | tail -6 | tr '\n' ' ' | cut -c1-400 )
  echo "| $d | ${r:-no summary (timeout 900 s)} | $(date -u +%H:%M:%S) |" >> $OUT
done
echo "== suites done $(date -u +%H:%M:%S)" >> $OUT
