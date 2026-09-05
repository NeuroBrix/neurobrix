#!/bin/bash
cd /home/mlops/NeuroBrix_System; PY=/home/mlops/ml/venv/bin/python; D=2026_09_05_image_final2
export NBX_SOCKET_PATH=/home/mlops/.neurobrix/daemon_gpu3.sock
$PY benchmarks/harness/run_bench.py --row image_diffusion_pixart_sigma --columns neurobrix_pytorch,diffusers --gpu 3 --date $D --repetitions 5 --env NBX_STEP_CACHE_THRESHOLD= --env NBX_STEP_CACHE_MAX_SKIPS= > benchmarks/results/$D/run_sigma.out 2>&1
$PY benchmarks/harness/run_bench.py --row image_diffusion_pixart_xl --columns neurobrix_pytorch,diffusers --gpu 3 --date $D --repetitions 5 --env NBX_STEP_CACHE_THRESHOLD= --env NBX_STEP_CACHE_MAX_SKIPS= > benchmarks/results/$D/run_xl.out 2>&1
echo FINAL_DONE >> benchmarks/results/$D/run_xl.out
