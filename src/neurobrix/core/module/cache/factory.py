"""
StateCacheFactory — Declarative KV Cache Creation

ZERO HARDCODE: All config from Prism plan or defaults.json lm_config.
ZERO SEMANTIC: Pure cache allocation — no model-specific knowledge.

Consolidates the two creation paths:
1. Prism KVCachePlan (preferred — precomputed budget trade-offs)
2. Legacy defaults.json lm_config (fallback)

Usage:
    wrapper = StateCacheFactory.create(ctx, lm_name, device, dtype)
"""

import torch
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from neurobrix.core.flow.base import FlowContext
    from neurobrix.core.runtime.graph.kv_cache_wrapper import KVCacheAttentionWrapper


class StateCacheFactory:
    """
    Factory for creating KV cache wrappers from runtime data.

    Prism-first: Uses KVCachePlan when available (optimal allocation).
    Legacy fallback: Derives from defaults.json lm_config.
    """

    @staticmethod
    def create(
        ctx: 'FlowContext',
        lm_config: Dict[str, Any],
        device: str,
        dtype: torch.dtype,
    ) -> 'KVCacheAttentionWrapper':
        """
        Create KV cache wrapper using optimal path.

        Priority:
        1. Prism KVCachePlan (precomputed budget trade-offs)
        2. Legacy defaults.json lm_config

        Args:
            ctx: FlowContext with plan and package data
            lm_config: LM configuration from defaults.json
            device: Target device (ignored — lazy allocation)
            dtype: Execution dtype

        Returns:
            Configured KVCacheAttentionWrapper

        ZERO FALLBACK: Missing critical data raises explicit error.
        """
        from neurobrix.core.runtime.graph.kv_cache_wrapper import (
            KVCacheConfig, KVCacheAttentionWrapper, create_kv_wrapper_from_config
        )

        # Validate required keys
        for key in ["num_layers", "num_heads", "hidden_size"]:
            if not lm_config.get(key):
                raise RuntimeError(
                    f"ZERO FALLBACK: '{key}' not found in lm_config.\n"
                    "Model data incomplete. Re-import: neurobrix remove <model> && neurobrix import <org>/<model>"
                )

        # Path 1: Prism KVCachePlan (PREFERRED)
        kv_plan = getattr(ctx.plan, 'kv_cache_plan', None)
        if kv_plan is not None:
            config = KVCacheConfig(
                num_layers=kv_plan.num_layers,
                num_kv_heads=kv_plan.num_kv_heads,
                k_head_dim=kv_plan.k_head_dim,
                v_head_dim=kv_plan.v_head_dim,
                max_cache_len=kv_plan.max_cache_len,
                dtype=kv_plan.dtype,
            )
            wrapper = KVCacheAttentionWrapper(config)
            return wrapper

        # Path 2: Legacy fallback
        wrapper = create_kv_wrapper_from_config(lm_config, device, dtype)
        return wrapper
