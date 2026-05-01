"""
InferenceEngine — Persistent model serving for NeuroBrix.

Load once, serve many. Weights stay in VRAM between requests.
Works for ALL families: LLM, image, audio, video.

Strategy-aware:
  Warm strategies (single_gpu eager, pipeline, tp, fgp):
    Weights persist in VRAM across requests. Near-zero latency after first request.
  Cold strategies (lazy, zero3, pp_lazy):
    Weights reload per request (model too large for VRAM). Setup + context still persist.

ZERO SEMANTIC: Family-aware only for output formatting, not for execution.
ZERO HARDCODE: All config from NBX container + Prism hardware profile.
"""

import time
import json
import torch
from neurobrix.core.device_utils import device_empty_cache, device_sync, device_seed, device_memory_stats
from pathlib import Path
from typing import Any, Dict, Optional

from neurobrix.serving.session import ConversationSession


# Warm serving is now determined by loading_mode from Prism solver.
# loading_mode="eager" → weights stay in VRAM (warm serving)
# loading_mode="lazy"  → weights reload per request (cold serving)
# No manual strategy set needed — Prism is the single source of truth.


class InferenceEngine:
    """
    Persistent inference engine — load once, serve many.

    Wraps RuntimeExecutor and keeps it alive across requests.
    For warm strategies: GraphExecutor weights persist in VRAM via _persistent flag.
    For cold strategies: weights reload per request but setup is reused.

    For LLM models: manages ConversationSession for multi-turn chat.
    For image/audio/video: each generate() call is independent.
    """

    def __init__(self, model_name: str, hardware_id: str, mode: str = "compiled"):
        self.model_name = model_name
        self.hardware_id = hardware_id
        self.mode = mode

        self._pkg = None           # RuntimePackage (immutable)
        self._plan = None          # ExecutionPlan (immutable)
        self._executor = None      # RuntimeExecutor (persistent, setup once)
        self._container = None     # NBXContainer
        self._family = None        # "llm", "image", "audio", "video"
        self._session = None       # ConversationSession (LLM only)
        self._is_loaded = False
        self._warm_serving = False  # True if strategy supports persistent weights
        self._device = None        # Primary device string (set during load)

    def load(self) -> None:
        """
        One-time setup: container -> Prism -> RuntimeExecutor.setup().

        After this call, the engine is ready to serve requests.
        First request loads weights into VRAM. For warm strategies,
        weights persist across all subsequent requests.
        """
        from neurobrix.nbx import NBXContainer
        from neurobrix.core.prism import PrismSolver, load_profile, InputConfig
        from neurobrix.core.runtime.loader import NBXRuntimeLoader
        from neurobrix.core.runtime.executor import RuntimeExecutor
        from neurobrix.cli.utils import find_model

        t_start = time.time()

        # 1. Find model
        nbx_path = find_model(self.model_name)

        # 2. Load container
        self._container = NBXContainer.load(str(nbx_path))
        manifest = self._container.get_manifest() or {}
        self._family = manifest.get("family")
        neural_components = self._container.get_neural_components()

        # 3. Solve Prism allocation
        hw_profile = load_profile(self.hardware_id)

        cache_path = self._container._cache_path
        defaults_path = cache_path / "runtime" / "defaults.json"
        cached_defaults = {}
        if defaults_path.exists():
            with open(defaults_path) as f:
                cached_defaults = json.load(f)

        height = cached_defaults.get("height", 1024)
        width = cached_defaults.get("width", 1024)
        vae_scale = cached_defaults.get("vae_scale_factor", 8)

        input_config = InputConfig(
            batch_size=2,
            height=height,
            width=width,
            dtype="float16",
            vae_scale=vae_scale,
        )

        solver = PrismSolver()
        self._plan = solver.solve_smart(self._container, hw_profile, input_config, serve_mode=True)

        # 3b. Resolve primary device from plan
        self._device = self._plan.primary_device

        # 3c. Apply CPU optimizations from hardware profile
        if hw_profile.cpu:
            from neurobrix.core.prism.cpu_config import apply_cpu_config
            apply_cpu_config(
                cpu=hw_profile.cpu,
                strategy=self._plan.strategy,
                device_count=hw_profile.device_count,
                preferred_dtype=hw_profile.preferred_dtype,
            )

        # 4. Determine warm serving compatibility
        # loading_mode is set by Prism solver based on strategy type:
        #   eager = weights stay in VRAM (single_gpu, pp, fgp, tp)
        #   lazy  = weights reload per request (pp_lazy, lazy_sequential, zero3)
        loading_mode = getattr(self._plan, 'loading_mode', 'lazy')
        self._warm_serving = (loading_mode == "eager")

        # 5. Load RuntimePackage
        loader = NBXRuntimeLoader()
        self._pkg = loader.load(str(nbx_path))

        # 6. Create RuntimeExecutor and run one-time setup
        # Data-driven hardware capability surface for Triton kernel wrappers.
        from neurobrix.kernels.wrappers import set_hardware_profile
        set_hardware_profile(hw_profile)
        self._executor = RuntimeExecutor(self._pkg, self._plan, mode=self.mode)

        # Set persistent mode BEFORE setup/execute so GraphExecutors get the flag
        if self._warm_serving:
            self._executor._persistent_mode = True

        self._executor.setup()

        # 7. Pre-warm weights for warm strategies (load into VRAM now, not on R1)
        # Only LLM family benefits from a tiny text warmup. Other families
        # (tts, stt, audio_llm, vlm, multimodal, image, upscaler, video) need
        # real-modality inputs (audio, image, etc.) or have multi-step paths
        # that don't respond to "warmup" + max_tokens=1 — skip warmup for them.
        from neurobrix.core.runtime.output_dispatch import family_uses_text_warmup
        if self._warm_serving and family_uses_text_warmup(self._family or ""):
            warmup_inputs = {
                "global.prompt": "warmup",
                "global.max_tokens": 1,
                "global.chat_mode": True,
            }
            self._executor.execute(warmup_inputs)

        t_load = time.time() - t_start
        self._is_loaded = True

        print(f"[NeuroBrix] {self.model_name} | {self.hardware_id} | {self._plan.strategy} | Ready in {t_load:.2f}s")

        if not self._warm_serving:
            print(f"[NeuroBrix] Cold serving — weights reload per request, setup/context persists")

    def generate(self, prompt: str = "", **kwargs) -> Dict[str, Any]:
        """
        Single-shot generation. Reused executor, fresh variable resolution.

        Works for ALL families. Returns structured result dict.
        """
        if not self._is_loaded:
            raise RuntimeError("ZERO FALLBACK: Engine not loaded. Call load() first.")

        # Build inputs dict (same structure as cmd_run)
        inputs = {}
        if prompt:
            inputs["global.prompt"] = prompt
        if "audio_path" in kwargs and kwargs["audio_path"]:
            inputs["global.audio_path"] = kwargs.pop("audio_path")
        if "steps" in kwargs and kwargs["steps"] is not None:
            inputs["global.num_inference_steps"] = kwargs["steps"]
        if "height" in kwargs and kwargs["height"] is not None:
            inputs["global.height"] = kwargs["height"]
        if "width" in kwargs and kwargs["width"] is not None:
            inputs["global.width"] = kwargs["width"]
        if "cfg" in kwargs and kwargs["cfg"] is not None:
            inputs["global.guidance_scale"] = kwargs["cfg"]
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            inputs["global.temperature"] = kwargs["temperature"]
        if "repetition_penalty" in kwargs and kwargs["repetition_penalty"] is not None:
            inputs["global.repetition_penalty"] = kwargs["repetition_penalty"]
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            inputs["global.max_tokens"] = kwargs["max_tokens"]
        if "chat_mode" in kwargs:
            inputs["global.chat_mode"] = kwargs["chat_mode"]
        if "seed" in kwargs and kwargs["seed"] is not None:
            seed = kwargs["seed"]
            inputs["global.seed"] = seed
            device_seed(self._device, seed)

        # Execute
        t_start = time.time()
        device_sync(self._device)

        outputs = self._executor.execute(inputs)

        device_sync(self._device)
        t_total = time.time() - t_start

        # Build result — DATA-DRIVEN by family output_format
        from neurobrix.core.runtime.output_dispatch import extract_result
        result = {"timing": {"total_s": round(t_total, 3)}, "family": self._family}
        result.update(extract_result(outputs, self._family or "", self._executor))
        return result

    def _generate_from_token_ids(self, token_ids: list, **kwargs) -> Dict[str, Any]:
        """
        Generate from pre-tokenized token IDs (chat path).

        Bypasses TextProcessor entirely — token IDs already include chat template,
        special tokens, and generation prompt. No double-BOS possible.
        """
        if not self._is_loaded:
            raise RuntimeError("ZERO FALLBACK: Engine not loaded. Call load() first.")

        # Build inputs with pre-tokenized IDs instead of string prompt
        inputs = {
            "global.input_token_ids": token_ids,
            "global.chat_mode": False,  # Template already applied
        }
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            inputs["global.temperature"] = kwargs["temperature"]
        if "repetition_penalty" in kwargs and kwargs["repetition_penalty"] is not None:
            inputs["global.repetition_penalty"] = kwargs["repetition_penalty"]
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            inputs["global.max_tokens"] = kwargs["max_tokens"]
        if "seed" in kwargs and kwargs["seed"] is not None:
            seed = kwargs["seed"]
            inputs["global.seed"] = seed
            device_seed(self._device, seed)

        # Execute
        t_start = time.time()
        device_sync(self._device)

        outputs = self._executor.execute(inputs)

        device_sync(self._device)
        t_total = time.time() - t_start

        result = {"timing": {"total_s": round(t_total, 3)}, "family": self._family}
        result.update(self._extract_llm_output(outputs))
        return result

    def chat(self, message: str, **kwargs) -> str:
        """
        Multi-turn chat for LLM models. Manages conversation session.

        Builds full context from conversation history each turn.
        Context overflow protection: summarizes old turns if KV cache limit approached.
        """
        if self._family != "llm":
            raise RuntimeError(
                f"ZERO FALLBACK: chat() requires LLM family, got '{self._family}'."
            )

        if self._session is None:
            tokenizer = self._executor.modules.get("tokenizer")
            if tokenizer is None:
                raise RuntimeError("ZERO FALLBACK: No tokenizer module in executor.")
            self._session = ConversationSession(
                tokenizer=tokenizer,
                defaults=self._pkg.defaults,
            )

        # Add user message to history
        self._session.add_user_message(message)

        # Context overflow protection
        max_cache_len = self._get_max_cache_len()
        max_tokens = kwargs.get("max_tokens") or self._pkg.defaults.get("max_tokens", 512)
        summarized = self._session.ensure_fits(
            max_cache_len=max_cache_len,
            max_tokens=max_tokens,
            generate_fn=self.generate,
        )
        if summarized:
            print("[Engine] Conversation history compressed — older turns summarized")

        # Build pre-tokenized IDs from conversation history via chat_template.
        # This avoids double-BOS: the template already includes special tokens,
        # so we pass token IDs directly to the executor, bypassing TextProcessor.
        token_ids = self._session.build_token_ids()
        result = self._generate_from_token_ids(token_ids, **kwargs)

        # Extract response text
        response_text = result.get("text", "")
        self._session.add_assistant_message(response_text)

        return response_text

    def _get_max_cache_len(self) -> int:
        """Get KV cache capacity from Prism plan."""
        kv_plan = getattr(self._plan, 'kv_cache_plan', None)
        if kv_plan is not None:
            return kv_plan.max_cache_len
        # Fallback to max_position_embeddings from lm_config
        lm_config = self._pkg.defaults.get("lm_config", {})
        return lm_config.get("max_position_embeddings", 2048)

    def new_conversation(self) -> None:
        """Start a new conversation, clearing history."""
        if self._session is not None:
            self._session.clear()

    def unload(self) -> None:
        """Release all VRAM. Engine can be re-loaded later."""
        # Clear persistent flags so weights can actually be freed
        if self._executor is not None:
            for _, executor in self._executor.executors.items():
                if hasattr(executor, '_persistent'):
                    executor._persistent = False

            for name in list(self._executor.executors.keys()):
                self._executor._unload_component_weights(name)

        device_empty_cache(self._device)

        self._is_loaded = False

    def get_status(self) -> Dict[str, Any]:
        """Return engine status info."""
        status = {
            "model": self.model_name,
            "hardware": self.hardware_id,
            "family": self._family,
            "loaded": self._is_loaded,
            "mode": self.mode,
            "warm_serving": self._warm_serving,
        }

        if self._plan is not None:
            status["strategy"] = self._plan.strategy

        if self._is_loaded and self._device:
            stats = device_memory_stats(self._device)
            if stats["allocated_mb"] > 0:
                status["vram_used_gb"] = round(stats["allocated_mb"] / 1024, 2)

        if self._session is not None:
            status["context"] = self._session.get_context_info()

        return status

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def family(self) -> Optional[str]:
        return self._family

    def save_output(self, outputs: Dict[str, Any], output_path: str) -> str:
        """
        Save non-text output (image/audio/video) to file. Returns saved path.
        Delegates to data-driven output_dispatch.save_output.
        """
        from neurobrix.core.runtime.output_dispatch import save_output
        # Strip any "waveform" alias so save_output finds global.output_audio.
        wf = outputs.get("waveform")
        if wf is not None and "global.output_audio" not in outputs:
            outputs = {**outputs, "global.output_audio": wf}
        tx = outputs.get("transcription")
        if tx is not None and "global.transcription" not in outputs:
            outputs = {**outputs, "global.transcription": tx}
        return save_output(
            outputs, output_path, self._family or "", self._executor, self._pkg
        )

    def _extract_llm_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from LLM generation outputs."""
        output_tokens = outputs.get("output_tokens")
        if output_tokens is None:
            output_tokens = outputs.get("global.output_tokens")
        if output_tokens is None:
            return {"text": "", "tokens": 0}

        token_ids = output_tokens.flatten().tolist()
        text = ""

        tokenizer = self._executor.modules.get("tokenizer")
        if tokenizer is not None and hasattr(tokenizer, 'decode'):
            text = tokenizer.decode(token_ids, skip_special_tokens=True)

        return {
            "text": text,
            "tokens": len(token_ids),
        }
