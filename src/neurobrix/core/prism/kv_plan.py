"""The KV cache a plan carries — Prism's decision, the runtime's instruction (moved out of core/prism/solver.py on
2026-11-12, a pure move with its two serialisations: the plan's record and its reading back)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class KVCachePlan:
    """KV cache allocation computed by Prism. Runtime MUST use these values."""
    max_cache_len: int          # Max tokens cacheable — ceiling (Prism VRAM budget)
    num_layers: int
    num_kv_heads: int
    k_head_dim: int             # K head dimension (qk_nope + qk_rope for MLA, else head_dim)
    v_head_dim: int             # V head dimension (v_head_dim for MLA, else head_dim)
    dtype: str                   # Cache dtype string (may differ from weight dtype)
    memory_bytes: int           # Total KV cache memory budget (at max_cache_len)
    per_token_bytes: int        # Memory per cached token (for runtime info)
    initial_cache_len: int = 0  # Initial buffer size (0 = allocate at max_cache_len)
    # The positions a SERVING session would hold on this plan's device: the remaining room this cache was sized from, in
    # the declared window (`kv_cache_size(serve=True)`) — the longest context a user's request can reach on this memory
    # class, which the derived census keys a decoder's prompt lengths up to. None: the decoder's state declares no window
    # (no session is planned for it) or its floor does not fit the room.
    session_reach: Optional[int] = None
    # The rows of one prefill chunk (docs/design/prism_kv.md §5bis): a request's prefill split into chunks of exactly
    # this many rows, a multiple of the cache page, the last chunk completed with masked rows the next chunk or the
    # decode overwrites — so every chunk meets the same kernel keys and a turn served from a session's cache gives the
    # bytes of the same turns prefilled together. Chosen by Prism from the certified times of each candidate's census
    # keys (core/prism/prefill_chunk.py), never a constant; None: one chunk of the turn's own rows, the reason stated.
    # A serving session's turns read it; a one-shot request prefills in one pass.
    prefill_chunk: Optional[int] = None
    # Why prefill_chunk is what it is (core/prism/prefill_chunk.py): the candidates' certified costs, or why none was
    # eligible — stated, never silent.
    prefill_chunk_reason: str = ""
    # The math route's scores budget a served chunk's attention over the cached keys routes on (core/prism/
    # served_attention.py): 0 where every decoder card's matrix unit runs the flash route at the cache's head dim — the
    # flash writes no scores, so the session holds no reserve for them; None: the executing device's own budget.
    chunk_scores_budget_bytes: Optional[int] = None
    # Whether a turn's last chunk is completed to the chunk's rows. True for a chunk chosen from certified times (its
    # keys are the certified ones, and a turn's bytes do not depend on where it ended); False where no certified time
    # chose it: the last chunk runs its own rows, so a turn that fits the priced rows is ONE unpadded pass — padded to
    # 128 rows, a 1 124-token first request took 51.6 s to its first token against 10.2 s in one pass, and a one-row
    # continuation 3-6 s against 1.1 s (measured 2026-11-12, Qwen3-Coder-30B-A3B, one V100-32GB).
    prefill_chunk_pads: bool = True

    @property
    def head_dim(self) -> int:
        """Backward compat for code that reads .head_dim"""
        return self.k_head_dim

    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 * 1024)

    def to_record(self) -> Dict[str, Any]:
        """The plan's record of this cache (`ExecutionPlan.to_dict`)."""
        return {
            "max_cache_len": self.max_cache_len,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "k_head_dim": self.k_head_dim,
            "v_head_dim": self.v_head_dim,
            "dtype": self.dtype,
            "memory_mb": self.memory_mb,
            "per_token_bytes": self.per_token_bytes,
            # exact, never through the float MB above (a restored plan lost them: the defect A's step 0 names)
            "memory_bytes": int(self.memory_bytes),
            "initial_cache_len": int(self.initial_cache_len),
            "session_reach": self.session_reach,
            "prefill_chunk": self.prefill_chunk,
            "prefill_chunk_reason": self.prefill_chunk_reason,
            "chunk_scores_budget_bytes": self.chunk_scores_budget_bytes,
            "prefill_chunk_pads": self.prefill_chunk_pads,
        }

    @classmethod
    def from_record(cls, kv_data: Dict[str, Any]) -> "KVCachePlan":
        """A cache read back from a plan's record (`ExecutionPlan.from_dict`)."""
        return cls(
            max_cache_len=kv_data["max_cache_len"],
            num_layers=kv_data["num_layers"],
            num_kv_heads=kv_data["num_kv_heads"],
            k_head_dim=kv_data.get("k_head_dim") or kv_data["head_dim"],
            v_head_dim=kv_data.get("v_head_dim") or kv_data["head_dim"],
            dtype=kv_data["dtype"],   # required, like num_layers above
            memory_bytes=int(kv_data["memory_bytes"]),
            per_token_bytes=kv_data.get("per_token_bytes", 0),
            initial_cache_len=int(kv_data["initial_cache_len"]),
            session_reach=kv_data.get("session_reach"),
            # a plan saved before the field planned one chunk: None is that plan's meaning
            prefill_chunk=kv_data.get("prefill_chunk"),
            prefill_chunk_reason=kv_data.get("prefill_chunk_reason", ""),
            # a plan saved before the field routed on the device's budget: None is that plan's meaning
            chunk_scores_budget_bytes=kv_data.get("chunk_scores_budget_bytes"),
            # a plan saved before the field padded every chunk: True is that plan's meaning
            prefill_chunk_pads=kv_data.get("prefill_chunk_pads", True),
        )
