"""
NeuroTax V4 - Tensor Name Normalization.

PRINCIPE: Strict Isomorphism - 1 Token -> 1 Token translation.
- Numeric anchors (block indices) are PRESERVED
- Functional tokens are TRANSLATED via SynonymRegistry
- Structure is NEVER modified
- ZERO FALLBACK: Unknown tokens cause explicit errors in strict mode

Example:
    "transformer_blocks.0.attn1.to_q.weight"
    -> "block.0.attn.query.weight"

    "model.layers.12.self_attn.q_proj.weight"
    -> "model.block.12.attn.query.weight"

V4 Changes:
- Extended SynonymRegistry with comprehensive patterns
- Added normalize_strict() with ZERO FALLBACK
- Added build_reverse_map() for graph.json normalization
- Support for LLM, Diffusion, MoE, Audio, Video models
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


class SynonymRegistry:
    """
    Central mapping from vendor terms to NeuroTax functions.
    ZERO HARDCODE on model names - only structural patterns.

    Covers: LLM (GPT, LLaMA, Mistral), Diffusion (Flux, SDXL, PixArt),
            MoE (Mixtral), Audio (Whisper), Video, T5, CLIP, etc.
    """

    _REGISTRY: Dict[str, str] = {
        # ===========================================
        # TOPOLOGY - BLOCKS
        # ===========================================
        "transformer_blocks": "block",
        "single_transformer_blocks": "single_block",
        "layers": "block",
        "block": "block",
        "blocks": "block",
        "h": "block",                    # GPT-2 style
        "layer": "block",

        # Encoder/Decoder
        "encoder": "encoder",
        "decoder": "decoder",
        "model": "model",

        # ===========================================
        # TOPOLOGY - UNET/DIFFUSION
        # ===========================================
        "down_blocks": "down",
        "downsamplers": "down_sample",
        "up_blocks": "up",
        "upsamplers": "up_sample",
        "mid_block": "mid",
        "middle_block": "mid",
        "resnets": "resnet",
        "attentions": "attn",

        # ===========================================
        # ATTENTION
        # ===========================================
        "attn": "attn",
        "attn1": "self_attn",
        "attn2": "cross_attn",
        "attention": "attn",
        "self_attn": "attn",
        "self_attention": "attn",
        "cross_attn": "cross_attn",
        "cross_attention": "cross_attn",

        # Q/K/V projections
        "to_q": "query",
        "to_k": "key",
        "to_v": "value",
        "to_out": "out",
        "query": "query",
        "key": "key",
        "value": "value",
        "q_proj": "query",
        "k_proj": "key",
        "v_proj": "value",
        "o_proj": "out",
        "out_proj": "out",
        "qkv": "qkv",
        "qkv_proj": "qkv",

        # Additional attention (Flux joint attention)
        "add_q_proj": "add_query",
        "add_k_proj": "add_key",
        "add_v_proj": "add_value",
        "to_add_out": "add_out",

        # Attention norms (RMSNorm for Q/K)
        "norm_q": "norm_q",
        "norm_k": "norm_k",
        "norm_added_q": "norm_add_q",
        "norm_added_k": "norm_add_k",

        # ===========================================
        # FFN / MLP
        # ===========================================
        "ff": "ffn",
        "ffn": "ffn",
        "mlp": "ffn",
        "feed_forward": "ffn",
        "ff_context": "ffn_context",

        # FFN layers
        "fc1": "up",
        "fc2": "down",
        "up_proj": "up",
        "down_proj": "down",
        "gate_proj": "gate",
        "w1": "up",
        "w2": "down",
        "w3": "gate",
        "c_fc": "up",                    # GPT-2
        "c_proj": "down",                # GPT-2

        # GEGLU/SwiGLU patterns
        "net": "net",
        "proj_mlp": "proj_mlp",

        # Linear
        "linear": "proj",
        "linear_1": "proj_1",
        "linear_2": "proj_2",
        "fc": "proj",
        "dense": "proj",

        # ===========================================
        # NORMALIZATION
        # ===========================================
        "norm": "norm",
        "norm1": "norm1",
        "norm2": "norm2",
        "norm3": "norm3",
        "norm1_context": "norm1_ctx",
        "layer_norm": "norm",
        "layernorm": "norm",
        "ln": "norm",
        "ln_1": "ln_1",
        "ln_2": "ln_2",
        "ln_f": "ln_final",
        "group_norm": "gnorm",
        "rms_norm": "rmsnorm",
        "rmsnorm": "rmsnorm",

        # Specific norms
        "input_layernorm": "input_norm",
        "post_attention_layernorm": "post_attn_norm",
        "pre_feedforward_layernorm": "pre_ffn_norm",
        "final_layer_norm": "final_norm",
        "norm_out": "norm_out",

        # ===========================================
        # EMBEDDINGS
        # ===========================================
        "embed": "embed",
        "embedding": "embed",
        "embeddings": "embed",
        "embed_tokens": "token_embed",
        "wte": "token_embed",
        "wpe": "pos_embed",
        "rotary_emb": "rotary_embed",

        # Diffusion embeddings
        "x_embedder": "x_embed",
        "x_embed": "x_embed",
        "t_embedder": "t_embed",
        "y_embedder": "y_embed",
        "context_embedder": "context_embedder",  # Keep as-is (global)
        "time_embedding": "time_embed",
        "time_embed": "time_embed",
        "pos_embed": "pos_embed",
        "patch_embed": "patch_embed",
        "position_embedding": "pos_embed",

        # Timestep/Guidance
        "time_text_embed": "time_text_embed",
        "timestep_embedder": "time",
        "guidance_embedder": "guidance",
        "text_embedder": "text_embed",

        # ===========================================
        # MODULATION / CONDITIONING
        # ===========================================
        "adaln_single": "adaln",
        "adaln": "adaln",
        "adaLN_modulation": "adaln_mod",
        "emb": "emb",
        "scale_shift_table": "scale_shift",
        "caption_projection": "caption_proj",

        # ===========================================
        # CONVOLUTION
        # ===========================================
        "conv": "conv",
        "conv1": "conv1",
        "conv2": "conv2",
        "conv_in": "conv_in",
        "conv_out": "conv_out",
        "proj_in": "proj_in",
        "proj_out": "proj_out",
        "proj": "proj",
        "projection": "proj",

        # ===========================================
        # OUTPUT / HEAD
        # ===========================================
        "lm_head": "head",
        "head": "head",
        "logits": "logits",
        "output": "out",
        "out": "out",

        # ===========================================
        # MOE (Mixture of Experts)
        # ===========================================
        "experts": "expert",
        "expert": "expert",
        "gate": "router",
        "router": "router",
        "shared_expert": "shared_expert",
        "shared_expert_gate": "shared_router",

        # ===========================================
        # ENCODERS (CLIP/T5)
        # ===========================================
        "shared": "shared",
        "text_model": "text",
        "vision_model": "vision",
        "text_projection": "text_proj",
        "visual_projection": "vision_proj",
        "final_layer_norm": "final_norm",

        # T5
        "relative_attention_bias": "rel_attn_bias",
        "SelfAttention": "attn",
        "EncDecAttention": "cross_attn",
        "DenseReluDense": "ffn",
        "wi": "up",
        "wi_0": "up_0",
        "wi_1": "up_1",
        "wo": "down",

        # ===========================================
        # QUANTIZATION (VAE)
        # ===========================================
        "quant_conv": "quant_conv",
        "post_quant_conv": "post_quant_conv",

        # ===========================================
        # PARAMETERS (PRESERVE)
        # ===========================================
        "weight": "weight",
        "bias": "bias",
        "scale": "scale",
        "shift": "shift",
        "gamma": "gamma",
        "beta": "beta",

        # ===========================================
        # BUFFERS / STATE
        # ===========================================
        "running_mean": "running_mean",
        "running_var": "running_var",
        "num_batches_tracked": "num_batches_tracked",
        "freqs_cis": "freqs_cis",
        "sin_cached": "sin_cached",
        "cos_cached": "cos_cached",
        "inv_freq": "inv_freq",

        # ===========================================
        # ACTIVATIONS
        # ===========================================
        "act_mlp": "act_mlp",
        "act": "act",
    }

    # Patterns that should NEVER be modified
    PRESERVE_PATTERNS = [
        r"^\d+$",           # Numeric indices (0, 1, 2, ...)
        r"^weight$",
        r"^bias$",
        r"^scale$",
        r"^shift$",
        r"^gamma$",
        r"^beta$",
        # Component names (already standard, no normalization needed)
        r"^transformer$",
        r"^vae$",
        r"^text_encoder$",
        r"^text_encoder_2$",
        r"^text_encoder_3$",
        r"^unet$",
        r"^model$",
        r"^encoder$",
        r"^decoder$",
        r"^tokenizer$",
        r"^scheduler$",
        r"^vocoder$",
        r"^safety_checker$",
        r"^image_encoder$",
        r"^feature_extractor$",
        # Module role names (already standard)
        r"^attn$",
        r"^mlp$",
        r"^ffn$",
        r"^norm$",
        r"^ln$",
        r"^embed$",
        r"^proj$",
        r"^head$",
        r"^lm_head$",
    ]

    @classmethod
    def resolve(cls, token: str) -> str:
        """
        Resolve a vendor token to NeuroTax standard.
        Returns original token if no mapping found (permissive mode).
        """
        # Check case-sensitive first, then lowercase
        if token in cls._REGISTRY:
            return cls._REGISTRY[token]
        lower = token.lower()
        return cls._REGISTRY.get(lower, token)

    @classmethod
    def resolve_strict(cls, token: str, original_name: str) -> str:
        """
        Resolve a vendor token with ZERO FALLBACK.
        Raises ValueError if token is unknown and not preserved.
        """
        # Check if should preserve
        for pattern in cls.PRESERVE_PATTERNS:
            if re.match(pattern, token):
                return token

        # Try to resolve
        if token in cls._REGISTRY:
            return cls._REGISTRY[token]
        lower = token.lower()
        if lower in cls._REGISTRY:
            return cls._REGISTRY[lower]

        # ZERO FALLBACK - unknown token
        raise ValueError(
            f"ZERO FALLBACK: Unknown token '{token}' in '{original_name}'.\n"
            f"Add mapping to SynonymRegistry._REGISTRY or PRESERVE_PATTERNS."
        )

    @classmethod
    def add_synonym(cls, vendor_term: str, neurotax_term: str):
        """Add a custom synonym mapping."""
        cls._REGISTRY[vendor_term] = neurotax_term

    @classmethod
    def has_mapping(cls, token: str) -> bool:
        """Check if a token has a mapping (case-insensitive)."""
        return token in cls._REGISTRY or token.lower() in cls._REGISTRY


@dataclass
class ParsedTensor:
    """Parsed tensor name with structure."""
    original: str
    normalized: str
    tokens: List[str]
    block_idx: Optional[int]
    is_global: bool
    category: str  # "block", "global", "embedding", etc.


class NeuroTaxParser:
    """
    NeuroTax V4 Parser - Strict Isomorphism.

    Rule: L(Key_Vendor) == L(Key_NeuroTax)
    - Preserves numeric indices as-is
    - Translates functional tokens via SynonymRegistry
    - NEVER reorders or restructures

    Modes:
    - parse() / normalize(): Permissive (unknown tokens pass through)
    - normalize_strict(): ZERO FALLBACK (unknown tokens cause errors)
    """

    # Patterns for block index extraction
    BLOCK_PATTERNS = [
        r"transformer_blocks\.(\d+)",
        r"single_transformer_blocks\.(\d+)",
        r"layers\.(\d+)",
        r"block\.(\d+)",
        r"blocks\.(\d+)",
        r"h\.(\d+)",                      # GPT-2
        r"layer\.(\d+)",
        r"encoder\.layers?\.(\d+)",
        r"encoder\.block\.(\d+)",
        r"decoder\.layers?\.(\d+)",
        r"decoder\.block\.(\d+)",
        r"down_blocks\.(\d+)",
        r"up_blocks\.(\d+)",
        r"resnets\.(\d+)",
        r"attentions\.(\d+)",
    ]

    # Global tensor prefixes (not in blocks)
    GLOBAL_PREFIXES = [
        "pos_embed", "patch_embed", "x_embedder", "t_embedder", "y_embedder",
        "x_embed", "context_embedder",
        "adaln_single", "adaln", "time_embed", "time_embedding",
        "time_text_embed", "caption_projection",
        "proj_in", "proj_out", "conv_in", "conv_out",
        "scale_shift_table", "norm_out", "final_layer_norm", "ln_f",
        "shared", "embed_tokens", "wte", "wpe", "lm_head",
        "text_projection", "visual_projection",
        "quant_conv", "post_quant_conv",
        "model",  # LLM prefix
        "running", "buffer", "freqs", "sin", "cos", "inv_freq",  # Buffer prefixes
    ]

    def __init__(self):
        self._block_pattern = re.compile("|".join(f"({p})" for p in self.BLOCK_PATTERNS))

    def parse(self, tensor_name: str) -> ParsedTensor:
        """
        Parse a tensor name into normalized NeuroTax format (permissive).

        Args:
            tensor_name: Original vendor tensor name

        Returns:
            ParsedTensor with normalized name and metadata
        """
        # Extract block index
        block_idx = self._extract_block_idx(tensor_name)

        # Determine if global
        is_global = block_idx is None and self._is_global_tensor(tensor_name)

        # Determine category
        category = self._categorize(tensor_name, block_idx, is_global)

        # Normalize (permissive)
        normalized, tokens = self._normalize(tensor_name)

        return ParsedTensor(
            original=tensor_name,
            normalized=normalized,
            tokens=tokens,
            block_idx=block_idx,
            is_global=is_global,
            category=category,
        )

    def normalize(self, tensor_name: str) -> str:
        """
        Normalize a tensor name (permissive mode).
        Unknown tokens pass through unchanged.
        """
        normalized, _ = self._normalize(tensor_name)
        return normalized

    def normalize_strict(self, tensor_name: str) -> str:
        """
        Normalize a tensor name with ZERO FALLBACK.
        Raises ValueError if any token is unknown and not preserved.
        """
        parts = tensor_name.split(".")
        normalized_parts = []

        for part in parts:
            norm = SynonymRegistry.resolve_strict(part, tensor_name)
            normalized_parts.append(norm)

        return ".".join(normalized_parts)

    def _extract_block_idx(self, name: str) -> Optional[int]:
        """Extract block index from tensor name."""
        for pattern in self.BLOCK_PATTERNS:
            match = re.search(pattern, name)
            if match:
                return int(match.group(1))
        return None

    def _is_global_tensor(self, name: str) -> bool:
        """Check if tensor is global (not in a block)."""
        first_part = name.split(".")[0]
        return first_part in self.GLOBAL_PREFIXES

    def _categorize(self, name: str, block_idx: Optional[int], is_global: bool) -> str:
        """Categorize tensor by function."""
        lower = name.lower()

        if block_idx is not None:
            return "block"

        if is_global:
            if "embed" in lower or "embedder" in lower:
                return "embedding"
            if "adaln" in lower or "time" in lower:
                return "modulation"
            if "norm" in lower:
                return "normalization"
            if "proj" in lower or "conv" in lower:
                return "projection"
            if "running" in lower or "freq" in lower or "buffer" in lower:
                return "buffer"
            return "global"

        # Fallback
        return "other"

    def _normalize(self, name: str) -> Tuple[str, List[str]]:
        """
        Normalize tensor name using SynonymRegistry (permissive).

        Strict Isomorphism: same number of tokens in, same number out.
        """
        parts = name.split(".")
        normalized_parts = []

        for part in parts:
            # Preserve numeric indices
            if part.isdigit():
                normalized_parts.append(part)
                continue

            # Translate via registry (permissive - unknown passes through)
            translated = SynonymRegistry.resolve(part)
            normalized_parts.append(translated)

        return ".".join(normalized_parts), normalized_parts

    def extract_all_block_indices(self, tensor_names: List[str]) -> List[int]:
        """Extract all unique block indices from tensor names."""
        indices = set()
        for name in tensor_names:
            idx = self._extract_block_idx(name)
            if idx is not None:
                indices.add(idx)
        return sorted(indices)

    def group_by_block(self, tensor_names: List[str]) -> Dict[int, List[str]]:
        """Group tensor names by block index."""
        groups: Dict[int, List[str]] = {}

        for name in tensor_names:
            idx = self._extract_block_idx(name)
            if idx is not None:
                if idx not in groups:
                    groups[idx] = []
                groups[idx].append(name)

        return groups

    def filter_global_tensors(self, tensor_names: List[str]) -> List[str]:
        """Return only global tensors (not in blocks)."""
        return [
            name for name in tensor_names
            if self._extract_block_idx(name) is None
        ]

    @staticmethod
    def build_reverse_map(forward_map: Dict[str, str]) -> Dict[str, str]:
        """
        Build reverse mapping: normalized -> original.

        Args:
            forward_map: Dict of {original_name: normalized_name}

        Returns:
            Dict of {normalized_name: original_name}
        """
        return {v: k for k, v in forward_map.items()}

    @staticmethod
    def normalize_parent_module(parent_module: str, forward_map: Optional[Dict[str, str]] = None) -> str:
        """
        Normalize a parent_module name from graph.json using the forward map.

        Tries to find the best match by looking for keys that start with parent_module.

        Args:
            parent_module: Original parent_module from graph.json
            forward_map: Optional Dict of {original_name: normalized_name} from neurotax_map.json

        Returns:
            Normalized parent_module name
        """
        if forward_map:
            # Direct match with .weight suffix
            weight_key = f"{parent_module}.weight"
            if weight_key in forward_map:
                # Remove .weight from the normalized name
                return forward_map[weight_key].rsplit(".weight", 1)[0]

            # Direct match with .bias suffix
            bias_key = f"{parent_module}.bias"
            if bias_key in forward_map:
                return forward_map[bias_key].rsplit(".bias", 1)[0]

        # Fallback: normalize token by token
        parser = NeuroTaxParser()
        return parser.normalize(parent_module)


def normalize_tensor_name(name: str) -> str:
    """
    Convenience function to normalize a single tensor name (permissive).

    Args:
        name: Original vendor tensor name

    Returns:
        Normalized NeuroTax name
    """
    parser = NeuroTaxParser()
    return parser.normalize(name)


def normalize_tensor_name_strict(name: str) -> str:
    """
    Convenience function to normalize a single tensor name (ZERO FALLBACK).

    Args:
        name: Original vendor tensor name

    Returns:
        Normalized NeuroTax name

    Raises:
        ValueError: If any token is unknown
    """
    parser = NeuroTaxParser()
    return parser.normalize_strict(name)
