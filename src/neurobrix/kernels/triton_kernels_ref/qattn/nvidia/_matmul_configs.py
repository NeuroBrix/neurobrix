# Based on the https://github.com/triton-lang/triton/blob/main/python/triton/ops/matmul_perf_model.py

from itertools import product
import heapq
import torch
import triton
from triton.runtime import driver


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def _num_stages_num_warps():
    num_stages_num_warps = [(1, 1), (1, 2), (1, 4)]
    num_stages_num_warps += [(s, w) for s, w in product(range(2, 9), [1, 2, 4, 8])]
    return num_stages_num_warps


def int8_configs():
    configs = []
    for num_stages, num_warps in _num_stages_num_warps():
        for block_m in [64, 128, 256]:
            for block_k in [64, 128, 256]:
                for block_n in [64, 128, 256]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
    return configs


def int8_dynamic_configs():
    configs = []
    for num_stages, num_warps in _num_stages_num_warps():
        for block_m in [16, 32, 64, 128, 256]:
            for block_k in [32, 64, 128, 256]:
                for block_n in [64, 128, 256]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, "SPLIT_K": 1},
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
                    for split_k in [2, 4, 8, 16]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, "SPLIT_K": split_k},
                                num_stages=num_stages,
                                num_warps=num_warps,
                                pre_hook=init_to_zero("C"),
                            )
                        )
    return configs


def early_config_prune(configs, named_args):
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability()
    # BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages
    dtsize = named_args["A"].element_size()

    # 1. make sure we have enough smem
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages = kw["BLOCK_M"], kw["BLOCK_N"], kw["BLOCK_K"], config.num_stages

        max_shared_memory = driver.utils.get_device_properties(device)["max_shared_mem"]
        required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory <= max_shared_memory:
            pruned_configs.append(config)
    configs = pruned_configs

    # group configs by (BLOCK_M,_N,_K, num_warps)
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            kw["BLOCK_K"],
            config.num_warps,
            config.num_stages,
        )

        key = (BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
        if key in configs_map:
            configs_map[key].append((config, num_stages))
        else:
            configs_map[key] = [(config, num_stages)]

    pruned_configs = []
    for k, v in configs_map.items():
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps = k
        if capability[0] >= 8:
            # compute cycles (only works for ampere GPUs)
            mmas = BLOCK_M * BLOCK_N * BLOCK_K / (16 * 8 * 16)
            mma_cycles = mmas / min(4, num_warps) * 8

            ldgsts_latency = 300  # Does this matter?
            optimal_num_stages = ldgsts_latency / mma_cycles

            # nearest stages, prefer large #stages
            nearest = heapq.nsmallest(
                2,
                v,
                key=lambda x: 10 + abs(x[1] - optimal_num_stages)
                if (x[1] - optimal_num_stages) < 0
                else x[1] - optimal_num_stages,
            )

            for n in nearest:
                pruned_configs.append(n[0])
        else:  # Volta & Turing only supports num_stages <= 2
            random_config = v[0][0]
            random_config.num_stages = 2
            pruned_configs.append(random_config)
    return pruned_configs
