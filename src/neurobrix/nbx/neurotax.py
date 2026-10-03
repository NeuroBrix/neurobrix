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

#: The one version of the neurotaxe (the owner, 2026-09-26: "NeuroTax 5.0, one version"). Forge
#: writes it in the manifest (`neurotax_version`); the runtime loader refuses by name a container
#: whose keys were written under another version — a key the parser no longer emits is a key no
#: reader will find (the MoE fusion's expert match, first of them).
#: 5.0: the parser is a fixed point on its own output — the SwiGLU gate is `ffn_gate` (was `gate`,
#: the vendors' router token), diffusers' `attn1` is `attn` (was `self_attn`).
#: 5.1 (2026-10-04): the vocabulary is complete over the catalogue — 362 tokens the 5.0 registry
#: refused (and 4 it added after the keys were written) are translated, so 22 749 keys already
#: emitted under 5.0 change: a new version (the neurotaxe's rule 6), every container rewritten once
#: in place by Forge `tools/neurotax_rename.py`.
NEUROTAX_VERSION = "5.1"


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
        "fast_layers": "fast_block",     # Fish-Speech dual AR fast transformer
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
        # diffusers' attn1 is the block's self-attention: the same function the LLM `self_attn`
        # names, so the same canonical token. It was `self_attn`, a VENDOR token the registry
        # itself maps to `attn` — the parser rewrote its own output (5 164 keys, 2026-09-26).
        "attn1": "attn",
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
        "wo": "down",                    # Fish-Speech/LLaMA attn output projection
        "wqkv": "wqkv",                 # Fish-Speech combined QKV
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
        # The SwiGLU gate. Its canonical token was `gate`, which vendors use for the MoE ROUTER
        # (`mlp.gate`, mapped to `router` below): the parser turned every SwiGLU gate it had
        # written into a router on a second pass (63 841 keys, 2026-09-26). `ffn_gate` is the name
        # GGUF gives the SwiGLU gate (its router is `ffn_gate_inp`); no vendor uses it for anything
        # else, and no key of the 49 containers carries it (census of 2026-09-26).
        "gate_proj": "ffn_gate",
        "w1": "ffn_gate",                # LLaMA SwiGLU: w1 = gate_proj
        "w2": "down",                    # LLaMA SwiGLU: w2 = down_proj
        "w3": "up",                      # LLaMA SwiGLU: w3 = up_proj
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
        # Codec-token embedding tables (generative-speech talkers: the
        # talker backbone's codec vocab table and the MTP predictor's
        # per-group ModuleList tables — Qwen3-Omni lineage; MiniCPM-o /
        # Ming talkers reuse the term).
        "codec_embedding": "codec_embed",

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
        # `shared` is the T5/UMT5/CLIP TIED input embedding: the model ties one
        # weight as both the encoder input embedding and (in full enc-dec T5) the
        # output head. By intrinsic identity it IS the canonical input token
        # embedding — same role as `embed_tokens` / `wte` — so it normalizes to
        # `token_embed` (NOT to itself). Structural vendor token, model-agnostic.
        # Was self-mapped (`shared`->`shared`), which left `shared.weight`
        # unbindable to the graph param `encoder.token_embed.weight` (Wan UMT5
        # text encoder; recurs on CogVideoX / Allegro / SANA-Video / Open-Sora).
        "shared": "token_embed",
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

        # ===========================================
        # AUDIO — NeMo Conformer / RNNT / TDT
        # ===========================================
        "pre_encode": "pre_encode",
        "dec_rnn": "rnn",
        "lstm": "lstm",
        "prediction": "prediction",
        "joint_net": "joint",
        "jointnet": "joint",
        "subsampling": "subsample",
        "conformer": "conformer",

        # ===========================================
        # AUDIO — Whisper
        # ===========================================
        "encoder_attn": "cross_attn",
        "encoder_attn_layer_norm": "cross_attn_norm",

        # ===========================================
        # AUDIO — Kokoro / StyleTTS
        # ===========================================
        "plbert": "bert",
        "istftnet": "vocoder",
        "duration_proj": "dur_proj",
        "pitch_proj": "pitch_proj",

        # ===========================================
        # AUDIO — Fish-speech / OpenAudio (DualAR)
        # ===========================================
        "fast_layers": "fast_block",
        "codebook": "codebook",
        "wq": "query",
        "wk": "key",
        "wv": "value",
        "wo": "down",                    # Fish-Speech: attn output proj = attn.down

        # ===========================================
        # AUDIO — Chatterbox
        # ===========================================
        "s3gen": "s3gen",
        "ve": "voice_encoder",
        "t3": "t3",

        # ===========================================
        # AUDIO — VibeVoice
        # ===========================================
        "acoustic_tokenizer": "acoustic_tok",
        "semantic_tokenizer": "semantic_tok",
        "acoustic_connector": "acoustic_proj",
        "semantic_connector": "semantic_proj",
        "prediction_head": "pred_head",
        "diffusion_head": "diff_head",

        # ===========================================
        # VIDEO — T5/UMT5 atomic projections (Wan / CogVideoX text encoders)
        # T5 names its attention projections with bare letters; map them to
        # the same canonical targets as to_q / q_proj.
        # ===========================================
        "q": "query",
        "k": "key",
        "v": "value",
        "o": "out",

        # ===========================================
        # VIDEO — CogVideoX (DiT + causal 3D VAE)
        # ===========================================
        "norm_final": "final_norm",
        "pos_embedding": "pos_embed",
        "conv_y": "scale_conv",          # SpatialNorm3D scale branch
        "conv_b": "shift_conv",          # SpatialNorm3D shift branch
        "norm_layer": "norm",            # SpatialNorm3D inner GroupNorm
        "conv_shortcut": "skip_conv",    # ResNet residual 1x1 conv

        # ===========================================
        # VIDEO — Wan (DiT + causal 3D VAE)
        # ===========================================
        "patch_embedding": "patch_embed",
        "condition_embedder": "cond_embed",
        "time_embedder": "time_embed",
        "time_proj": "time_proj",
        "resample": "resample",          # spatial up/down resampler (canonical)
        "time_conv": "temporal_conv",
        "to_qkv": "qkv",

        # ===========================================
        # VIDEO — SANA-Video (linear DiT GLUMB-Conv + Gemma text encoder)
        # ===========================================
        "conv_inverted": "expand_conv",      # MBConv inverted-bottleneck expand
        "conv_depth": "depthwise_conv",
        "conv_point": "pointwise_conv",
        "conv_temp": "temporal_conv",
        "caption_norm": "caption_norm",
        "rope": "rotary_embed",
        "freqs_cos": "freqs_cos",            # rope table buffers (canonical)
        "freqs_sin": "freqs_sin",
        "post_feedforward_layernorm": "post_ffn_norm",
        "original_inv_freq": "original_inv_freq",

        # VIDEO — VAE latent statistics buffers (SANA-Video / LTX-class)
        "latents_mean": "latents_mean",
        "latents_std": "latents_std",

        # ===========================================================================================
        # NeuroTax 5.1 — the vocabulary completed over the catalogue (2026-10-04). Every entry is a
        # vendor's spelling of a role; the canonical side is an existing token wherever one names
        # the role (family alignment, rule 9). Tokens with a glued index (`conv3`, `tdnnd12`,
        # `feed_forward1`) are not listed one by one: the stem is translated and the index kept
        # (`resolve` below). Census of the 49 cached containers: 362 refused tokens, 23 649 keys.
        # ===========================================================================================

        # --- repeated units and stages -----------------------------------------------------------
        "double_blocks": "block",            # Flux-lineage double-stream blocks (diffusers: transformer_blocks)
        "single_blocks": "single_block",     # Flux-lineage single-stream blocks (diffusers: single_transformer_blocks)
        "vace_blocks": "control_block",      # Wan-VACE control (context) blocks beside the main blocks
        "stages": "stage",                   # multi-scale stages (Swin v2, ConvNeXt-style codec encoders)
        "conv_blocks": "conv_block",         # taming/VQGAN encoder and decoder stages of conv blocks
        "residual_group": "res_group",       # SwinIR/HAT residual group inside a residual Swin block
        "body": "trunk",                     # ESRGAN RRDB trunk (BasicSR `body`, original `RRDB_trunk`)
        "xvector": "trunk",                  # CAM++ speaker-embedding trunk
        "rdb": "dense_block",                # ESRGAN residual dense block (rdb1..rdb3)
        "tdnnd": "dense_layer",              # CAM++ dense TDNN layer (tdnnd1..)
        "transit": "transition",             # CAM++ transition layer (transit1..)
        "encoders": "enc_layer",             # ESPnet/CosyVoice conformer layer list
        "up_encoders": "up_enc_layer",       # CosyVoice upsampling conformer layer list
        "encode": "enc_resnet",              # StyleTTS2/Kokoro decoder: its AdaIN residual encode block
        "decode": "dec_resnet",              # StyleTTS2/Kokoro decoder: its AdaIN residual decode blocks
        "mid_blocks": "bottleneck_block",    # Matcha-TTS U-Net middle blocks (`mid_block` is diffusers' vendor token)
        "resblocks": "resblock",             # HiFi-GAN / iSTFTNet multi-receptive-field residual blocks
        "source_resblocks": "source_resblock",   # HiFT/NSF source-branch residual blocks
        "noise_res": "source_resnet",        # iSTFTNet (Kokoro) source-branch residual blocks
        "res": "resnet",                     # taming/VQGAN residual block list
        "albert_layer_groups": "layer_group",
        "albert_layers": "shared_block",     # ALBERT's cross-layer shared block
        "final_block": "out_block",          # Matcha-TTS U-Net output block
        "block_in": "in_block",              # Mochi VAE decoder entry stage
        "block_out": "out_block",            # Mochi VAE decoder exit stage
        "pre_transformer": "pre_net",        # codec decoders: transformer before the upsampler
        "pre_module": "pre_net",
        "post_module": "post_net",
        "conv_block": "conv_block",          # HAT: the convolutional block beside window attention
        "cab": "chan_attn",                  # HAT channel-attention block
        "overlap_attn": "overlap_cross_attn",# HAT overlapping cross-window attention
        "mixer": "token_mixer",              # ConvNeXt-style token mixer (VibeVoice codec)
        "cam_layer": "ctx_mask",             # CAM++ context-aware masking
        "nonlinear": "bn_act",               # CAM++ BatchNorm + activation pair
        "out_nonlinear": "out_bn_act",
        "fsmn_block": "memory_conv",         # FSMN memory block (S3 tokenizer attention)
        "tdnn": "conv_in",                   # CAM++ first TDNN layer: the trunk's input convolution

        # --- model and sub-model prefixes ---------------------------------------------------------
        "swin2sr": "model",                  # HF base-model prefix (same role as `model`)
        "vision_tower": "vision",            # same role as `vision_model`
        "tfmr": "backbone",                  # a speech LM's transformer backbone
        "flow": "acoustic",                  # flow-matching acoustic model (tokens -> mel)
        "estimator": "denoiser",             # the flow's velocity estimator network
        "mel2wav": "vocoder",                # same role as `istftnet`
        "generator": "vocoder",              # iSTFTNet generator
        "speaker_encoder": "voice_encoder",  # same role as `ve`
        "cond_enc": "cond_encoder",
        "qformer": "resampler",              # query transformer resampling a sequence to queries
        "perceiver": "resampler",
        "connector": "bridge",               # a model connecting two others' token spaces
        "preprocessor": "frontend",          # audio feature front-end
        "featurizer": "mel",
        "f0_predictor": "pitch_predictor",
        "condnet": "trunk",
        "m_source": "harmonic_source",       # NSF harmonic source module
        "merger": "mm_proj",                 # vision patch merger into the LM width
        "merger_list": "mm_proj",
        "deepstack_merger_list": "layer_mm_proj",
        "audio_projector": "audio_proj",
        "semantic_quantizer": "semantic_quant",
        "quantizer": "quant",
        "quantizers": "quant",
        "quantize": "quant",
        "_codebook": "codebook",
        "project_down": "down",              # a codebook's input projection to its lower width
        "lstms": "lstm",
        "cnn": "conv",

        # --- attention --------------------------------------------------------------------------
        "self": "attn",                      # HF BERT-style `attention.self`
        "crossattention": "cross_attn",
        "img_attn": "attn",                  # Flux-lineage image-stream attention
        "txt_attn": "ctx_attn",              # Flux-lineage text-stream attention
        "query_key_value": "qkv",
        "to_qkv_multiscale": "qkv_multiscale",
        "to_kv": "kv",
        "kv_proj": "kv",
        "kv": "kv",
        "linear_q": "query",
        "linear_k": "key",
        "linear_v": "value",
        "linear_out": "out",
        "linear_pos": "pos_proj",            # relative-position projection (Transformer-XL lineage)
        "pos_bias_u": "pos_bias_u",
        "pos_bias_v": "pos_bias_v",
        "kv_a_proj_with_mqa": "kv_down",     # multi-head latent attention: KV down-projection
        "kv_a_layernorm": "kv_norm",
        "kv_b_proj": "kv_up",
        "q_norm": "norm_q",
        "k_norm": "norm_k",
        "query_norm": "norm_q",
        "key_norm": "norm_k",
        "ln_q": "norm_q",
        "ln_kv": "norm_kv",
        "attn_pool": "pool_attn",            # attention-pooling head
        "latent": "pool_query",              # its learned probe
        "pre_attention_query": "pool_query",
        "relative_position_bias_table": "rel_attn_bias",   # same role as T5's relative_attention_bias
        "relative_position_index": "rel_pos_index",
        "relative_position_index_SA": "rel_pos_index_sa",
        "relative_position_index_OCA": "rel_pos_index_oca",
        "continuous_position_bias_mlp": "rel_pos_mlp",
        "rel_pos_emb": "rel_pos_embed",
        "attn_mask": "attn_mask",
        "causal_mask": "causal_mask",
        "logit_scale": "logit_scale",

        # --- feed-forward -----------------------------------------------------------------------
        "feed_forward": "ffn",               # stem of feed_forward1/2 (listed above as a module)
        "intermediate": "ffn",               # HF BERT-style intermediate.dense
        "intermediate_query": "query_ffn",   # Q-Former query-token FFN
        "output_query": "query_ffn_out",
        "img_mlp": "ffn",
        "txt_mlp": "ffn_context",            # same role as diffusers' ff_context
        "v_mlp": "value_ffn",
        "block_sparse_moe": "ffn",           # the MoE feed-forward (Mixtral lineage)
        "shared_experts": "shared_expert",
        "gate_up_proj": "ffn_gate_up",       # fused SwiGLU gate + up
        "input_linear": "proj_in",
        "output_linear": "proj_out",
        "audio_gate": "audio_router",        # modality-specific MoE routers
        "image_gate": "image_router",
        "w_1": "up",                         # ESPnet position-wise FFN
        "w_2": "down",
        "linear_fc1": "up",
        "linear_fc2": "down",
        "ffn_output": "ffn_out",
        "ff": "ffn",

        # --- norms ------------------------------------------------------------------------------
        "batchnorm": "bnorm",
        "batch_norm": "bnorm",
        "bn": "bnorm",
        "layer_norm": "norm",
        "norms": "norm",
        "conv_norm_out": "norm_out",         # diffusers VAE GroupNorm before conv_out
        "self_attn_layer_norm": "pre_attn_norm",
        "norm_self_att": "pre_attn_norm",
        "attn_ln": "pre_attn_norm",
        "norm_mha": "pre_attn_norm",
        "layernorm_before": "pre_attn_norm",
        "attention_norm": "input_norm",      # LLaMA-original spelling of input_layernorm
        "ffn_norm": "pre_ffn_norm",
        "norm_feed_forward": "pre_ffn_norm",
        "mlp_ln": "pre_ffn_norm",
        "norm_ff": "pre_ffn_norm",
        "layernorm_after": "pre_ffn_norm",
        "norm_conv": "pre_conv_norm",
        "post_self_attn_layernorm": "attn_out_norm",
        "post_mlp_layernorm": "post_ffn_norm",
        "full_layer_layer_norm": "post_ffn_norm",
        "pre_layrnorm": "pre_norm",
        "pre_norm": "pre_norm",
        "post_layernorm": "post_norm",
        "ln_post": "post_norm",
        "post_norm": "post_norm",
        "after_norm": "final_norm",
        "post_projection_norm": "post_proj_norm",
        "post_conv_layernorm": "post_conv_norm",
        "adain": "adanorm",                  # adaptive instance norm (adain1/2)
        "self_attn_layer_scale": "attn_scale",
        "attention_layer_scale": "attn_scale",
        "mlp_layer_scale": "ffn_scale",
        "ffn_layer_scale": "ffn_scale",
        "ffn_gamma": "ffn_scale",

        # --- convolutions -----------------------------------------------------------------------
        "convs": "conv",                     # stem of convs1/convs2
        "conv2d": "conv",                    # stem of conv2d1..3
        "convolution": "conv",               # stem of convolution_0/1
        "conv_first": "conv_in",
        "first_convolution": "conv_in",
        "conv_pre": "conv_in",
        "conv_last": "conv_out",
        "final_convolution": "conv_out",
        "conv_post": "conv_out",
        "conv_after_body": "trunk_conv",
        "conv_body": "trunk_conv",
        "conv_before_upsample": "pre_up_conv",
        "conv_up": "up_sample_conv",         # ESRGAN: the conv after each nearest upsampling (conv_up1..)
        "up_conv": "pointwise_conv1",        # a conformer conv module's expanding pointwise conv (NeMo: pointwise_conv1)
        "down_conv": "pointwise_conv2",      # ... and its contracting one (NeMo: pointwise_conv2)
        "conv_hr": "hr_conv",
        "pointwise_conv": "pointwise_conv",
        "pwconv": "pointwise_conv",
        "dwconv": "depthwise_conv",
        "depth_conv": "depthwise_conv",
        "depthwise_conv": "depthwise_conv",
        "temp_convs": "temporal_conv",
        "temp_conv_in": "temporal_conv_in",
        "temp_conv_out": "temporal_conv_out",
        "temp_conv_up": "temporal_up_conv",
        "temp_convs_down": "temporal_down_conv",
        "shortcut": "skip_conv",
        "nin_shortcut": "skip_conv",
        "res_conv": "skip_conv",
        "conv1x1": "skip_conv",
        "convtr": "deconv",
        "pool": "pool_conv",                 # iSTFTNet's learned (transposed-conv) upsampling pool
        "pre_lookahead_layer": "lookahead_conv",
        "upsample": "up_sample",
        "upsample_layers": "up_block",       # a codec decoder's upsampling stages
        "ups": "up_sample",
        "up_layer": "up_sample",
        "downsample": "down_sample",
        "downsample_layers": "down_block",   # a codec encoder's downsampling stages
        "source_downs": "source_down",
        "noise_convs": "source_down",        # iSTFTNet's spelling of the source-branch downsamplers

        # --- linears and projections -------------------------------------------------------------
        "lin": "proj",
        "l_linear": "proj",
        "linear_layer": "proj",
        "linear_local": "local_proj",
        "in_layer": "proj_1",                # MLP embedder: same roles as linear_1 / linear_2
        "out_layer": "proj_2",
        "in_proj": "proj_in",
        "encoder_proj": "enc_proj",
        "enc": "enc_proj",                   # transducer joint: encoder-side projection
        "pred": "pred_proj",                 # transducer joint: predictor-side projection
        "spkr_enc": "spk_proj",
        "spk_embed_affine_layer": "spk_proj",
        "emotion_adv_fc": "emotion_proj",
        "embedding_hidden_mapping_in": "embed_proj",
        "output_mlp_projector": "head_proj",
        "vision_head": "vision_head",
        "text_head": "text_head",
        "speech_head": "speech_head",
        "classifier": "head",
        "out_mid": "mid_head",               # an intermediate (CTC) output head
        "final_proj": "proj_out",
        "cond_proj": "cond_proj",
        "noisy_images_proj": "x_embed",      # the denoiser's noisy-input projection
        "final_layer": "head",               # DiT final layer: adaLN + output projection
        "pooler": "pool_head",               # a pooling head (BERT pooler, attention pooler)
        "audio_proj": "audio_proj",

        # --- embeddings and conditioning ---------------------------------------------------------
        "token_embedding": "token_embed",
        "word_embeddings": "token_embed",
        "input_embedding": "token_embed",
        "code_embedding": "codec_embed",
        "codebook_embeddings": "codebook_embed",
        "fast_embeddings": "fast_embed",
        "fast_norm": "fast_norm",
        "fast_output": "fast_out",
        "text_emb": "text_token_embed",
        "speech_emb": "speech_token_embed",
        "text_pos_emb": "text_pos_embed",
        "speech_pos_emb": "speech_pos_embed",
        "position_embeddings": "pos_embed",
        "embed_positions": "pos_embed",
        "positional_embedding": "pos_embed",
        "patch_embeddings": "patch_embed",
        "token_type_embeddings": "type_embed",
        "class_embedding": "cls_embed",
        "position_ids": "pos_ids",
        "pos_frequencies": "pos_freqs",
        "ref_pos_embed": "ref_pos_embed",
        "image_embedder": "image_embed",
        "vace_patch_embedding": "control_patch_embed",
        "pos_embed_first_frame": "first_frame_embed",
        "pos_embed_mask": "mask_embed",
        "pos_embed_masked_video": "masked_video_embed",
        "img_in": "x_embed",                 # Flux lineage: same role as x_embedder
        "txt_in": "context_embedder",        # same role as context_embedder
        "time_in": "time_embed",
        "vector_in": "text_embed",           # pooled-text embedder (diffusers: text_embedder)
        "cond_in": "cond_embed",
        "time_mlp": "time_embed",
        "y_embedding": "uncond_embed",       # the caption projection's learned null embedding
        "aspect_ratio_embedder": "aspect_embed",
        "resolution_embedder": "res_embed",
        "up_embed": "up_input_embed",
        "img_mod": "mod",                    # Flux-lineage modulation
        "txt_mod": "ctx_mod",
        "modulation": "mod",
        "noise_refiner": "latent_refiner",
        "ref_image_refiner": "ref_refiner",

        # --- speech: pitch, energy, source ------------------------------------------------------
        "F0": "pitch",                       # StyleTTS2 lineage: fundamental-frequency branch
        "N": "energy",                       # StyleTTS2 lineage: energy branch
        "F0_conv": "pitch_conv",
        "N_conv": "energy_conv",
        "F0_proj": "pitch_proj",
        "N_proj": "energy_proj",
        "asr_res": "align_res",              # residual on the aligned text features

        # --- parameters and buffers -------------------------------------------------------------
        "alpha": "alpha",                    # Snake activation's learned frequency
        "activations": "act",                # stem of activations1/2
        "similarity_weight": "sim_weight",
        "similarity_bias": "sim_bias",
        "codebook_used": "codebook_usage",
        "_mel_filters": "mel_filters",
        "fb": "mel_filters",
        "window": "window",
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
        # Audio component names
        r"^joint$",
        r"^perception$",
        r"^audio_tower$",
        r"^multi_modal_projector$",
        r"^embed_tokens$",
        # LoRA wrapper tokens (preserve for suffix matching)
        r"^base_layer$",
        r"^base_model$",
        # PyTorch's own parameter spellings — the framework's names, the same class as `weight` /
        # `bias` / `running_mean`, never a vendor's: nn.RNN/LSTM/GRU per-layer parameters,
        # nn.MultiheadAttention's packed input projection, the legacy weight_norm pair and the
        # parametrize API (`<module>.parametrizations.weight.original0/1`).
        r"^(weight|bias)_(ih|hh)_l\d+(_reverse)?$",
        r"^in_proj_(weight|bias)$",
        r"^weight_[gv]$",
        r"^parametrizations$",
        r"^original\d+$",
    ]

    #: A token that carries its index glued to it (`conv3`, `tdnnd12`, `feed_forward1`, `linear_1`):
    #: the stem, the optional underscore, the index.
    _GLUED_INDEX = re.compile(r"^(.*?[A-Za-z])(_?)(\d+)$")

    @classmethod
    def _explicit(cls, token: str) -> Optional[str]:
        """The registry's own translation of a token (case-sensitive first, then lowercase)."""
        if token in cls._REGISTRY:
            return cls._REGISTRY[token]
        return cls._REGISTRY.get(token.lower())

    @classmethod
    def _lookup(cls, token: str) -> Optional[str]:
        """The translation of a token, or None when the registry does not know it.

        An explicit entry wins (`fc1` -> `up`, `norm1` -> `norm1`, `linear_1` -> `proj_1`). A token
        with a glued index whose stem the registry knows — as a vendor token or as a canonical one —
        is the stem's translation with the index kept as it is (`tdnnd12` -> `dense_layer12`,
        `conv3` -> `conv3`, `layer_norm1` -> `norm1`): an index is a numeric anchor, glued or not,
        and listing `tdnnd1` .. `tdnnd24` one by one would refuse the 25th. Canonical side: a
        stem's translation never ends in a digit, so the derived token is read back to itself.
        """
        hit = cls._explicit(token)
        if hit is not None:
            return hit
        m = cls._GLUED_INDEX.match(token)
        if m is None:
            return None
        stem, sep, idx = m.groups()
        canon = cls._explicit(stem)
        if canon is None and stem in cls.canonical_tokens():
            canon = stem
        if canon is None or canon[-1].isdigit():
            return None
        return f"{canon}{sep}{idx}"

    @classmethod
    def resolve(cls, token: str) -> str:
        """
        Resolve a vendor token to NeuroTax standard.
        Returns original token if no mapping found (permissive mode).
        """
        hit = cls._lookup(token)
        return token if hit is None else hit

    @classmethod
    def resolve_strict(cls, token: str, original_name: str) -> str:
        """
        Resolve a vendor token with ZERO FALLBACK.
        Raises ValueError if token is unknown and not preserved.
        """
        hit = cls._lookup(token)
        if hit is not None:
            return hit

        # A preserved token with no translation stays as it is. The registry is read FIRST: the
        # preserve list named tokens the registry translates (`mlp` -> `ffn`, `ln` -> `norm`,
        # `lm_head` -> `head`, `embed_tokens` -> `token_embed`), and strict kept them raw while
        # permissive translated them — two keys for one tensor.
        for pattern in cls.PRESERVE_PATTERNS:
            if re.match(pattern, token):
                return token

        # A canonical token the registry emits is placed: it is its own translation. Safe only
        # because every canonical token that is ALSO a registry key maps to itself (the fixed
        # point, `tests/unit/nbx/test_the_parser_is_a_fixed_point_on_its_own_output.py`) —
        # otherwise strict would answer the token and permissive its translation.
        if token in cls.canonical_tokens():
            return token

        # ZERO FALLBACK - unknown token
        raise ValueError(
            f"ZERO FALLBACK: Unknown token '{token}' in '{original_name}'.\n"
            f"Add mapping to SynonymRegistry._REGISTRY or PRESERVE_PATTERNS."
        )

    @classmethod
    def canonical_tokens(cls) -> frozenset:
        """Every token the registry translates TO."""
        return frozenset(cls._REGISTRY.values())

    @classmethod
    def add_synonym(cls, vendor_term: str, neurotax_term: str):
        """Add a custom synonym mapping."""
        cls._REGISTRY[vendor_term] = neurotax_term

    @classmethod
    def has_mapping(cls, token: str) -> bool:
        """Check if a token has a mapping (case-insensitive)."""
        return cls._lookup(token) is not None


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
