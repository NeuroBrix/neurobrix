"""
Flow Handler Base Classes

ZERO SEMANTIC: Flow handlers execute mechanically based on topology.json.
ZERO HARDCODE: All configuration comes from NBX container.

This module provides:
- FlowContext: Immutable context shared by all flow handlers
- FlowHandler: Abstract base class for execution flows
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from neurobrix.core.runtime.loader import RuntimePackage
    from neurobrix.core.runtime.resolution.variable_resolver import VariableResolver
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    from neurobrix.core.strategies import ExecutionStrategy


@dataclass
class FlowContext:
    """
    Immutable context passed to flow handlers.

    Contains all shared state needed for execution without
    requiring flow handlers to access executor internals.
    """
    # Core NBX package (immutable)
    pkg: 'RuntimePackage'

    # Prism execution plan with device allocations
    plan: Any

    # Variable resolver for dynamic binding
    variable_resolver: 'VariableResolver'

    # Component executors (GraphExecutor instances)
    executors: Dict[str, 'GraphExecutor']

    # Loaded modules (scheduler, tokenizer, etc.)
    modules: Dict[str, Any]

    # Execution strategy from Prism
    strategy: 'ExecutionStrategy'

    # Pre-indexed connections: {comp_name: {input_name: [sources]}}
    connections_index: Dict[str, Dict[str, List[str]]]

    # Loop identifier from variables contract
    loop_id: str

    # Path to NBX cache for weight loading
    nbx_path_str: str

    # Execution mode: "compiled" (default), "sequential" (PyTorch eager debug),
    # "triton" (Triton-pure compiled), "triton_sequential" (Triton-pure debug)
    mode: str = "compiled"

    # Persistent mode: when True, GraphExecutors mark themselves _persistent
    # so cleanup() preserves weights in VRAM. Set by serving layer for warm strategies.
    persistent_mode: bool = False

    # Primary device string — ALWAYS derived from Prism allocation (executor.py:_get_primary_device)
    primary_device: str = ""

    # Resolution binning (`resolution.resolution_binning.BinnedRequest`): set when the request
    # runs at a trained bin instead of its own size; the flow restores the requested size after
    # the decoder. None when the request is not binned.
    binned_request: Any = None

    def compute_dtype(self, component: str = None) -> 'torch.dtype':
        """Prism-RESOLVED compute dtype for flow-level tensor synthesis.

        SINGLE compiled-side resolver (brick-consolidation E2). The Prism
        plan is the authority for the dtype that actually executes — the
        manifest carries the pre-Prism vendor declaration and is only a
        last-resort fallback when no plan is attached (degraded/isolation
        contexts). Flow handlers that previously read `manifest["dtype"]`
        directly could diverge from the allocation the DtypeEngine runs
        under (e.g. a bf16 manifest resolved to fp32 on non-bf16 hardware).

        Resolution order:
          1. `plan.components[component].dtype` — per-component resolved
             dtype, when the caller names the component it synthesises for;
          2. first allocation carrying a dtype — the model-wide answer for
             single-dtype plans (every allocation agrees);
          3. `plan.target_dtype` — the plan-wide resolved dtype;
          4. `manifest["dtype"]` — no plan attached.

        The triton mirror is `neurobrix.triton.dtype.resolve_compute_dtype`
        (string-dtype boundary, R33) — separate implementation by design.
        """
        from neurobrix.core.dtype.config import get_torch_dtype
        plan = self.plan
        if plan is not None:
            comps = getattr(plan, "components", None)
            if comps:
                alloc = comps.get(component) if component else None
                if alloc is None or not getattr(alloc, "dtype", None):
                    alloc = next((a for a in comps.values()
                                  if getattr(a, "dtype", None)), None)
                if alloc is not None and getattr(alloc, "dtype", None):
                    return get_torch_dtype(alloc.dtype)
            target = getattr(plan, "target_dtype", None)
            if target:
                return get_torch_dtype(target)
        return get_torch_dtype(self.pkg.manifest.get("dtype", "float16"))


class FlowHandler(ABC):
    """
    Abstract base class for execution flow handlers.

    Each flow type (iterative_process, static_graph, forward_pass,
    autoregressive_generation) is implemented as a separate FlowHandler.

    ZERO SEMANTIC: FlowHandlers don't know about model semantics.
    They execute the flow mechanically based on topology.
    """

    def __init__(self, ctx: FlowContext):
        """
        Initialize flow handler with context.

        Args:
            ctx: FlowContext containing all execution state
        """
        self.ctx = ctx

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the flow and return final outputs.

        Returns:
            Dict of resolved variables/outputs from execution
        """
        pass



# Flow type registry for factory pattern
FLOW_REGISTRY: Dict[str, type] = {}

# flow type -> the module of its compiled (ATen) handler. Imported on demand
# by get_flow_handler; the Triton branch has its mirrors under
# neurobrix.triton.flow and never loads these (R33).
COMPILED_FLOW_MODULES: Dict[str, str] = {
    "iterative_process": "neurobrix.core.flow.iterative_process",
    "static_graph": "neurobrix.core.flow.static_graph",
    "forward_pass": "neurobrix.core.flow.forward_pass",
    "autoregressive_generation": "neurobrix.core.flow.autoregressive",
    "audio": "neurobrix.core.flow.audio",
    "encoder_decoder": "neurobrix.core.flow.encoder_decoder",
    "audio_llm": "neurobrix.core.flow.audio_llm",
    "vlm": "neurobrix.core.flow.vlm",
    "dual_ar": "neurobrix.core.flow.dual_ar",
    "tts_llm": "neurobrix.core.flow.tts_llm",
    "next_token_diffusion": "neurobrix.core.flow.next_token_diffusion",
    "rnnt": "neurobrix.core.flow.rnnt",
}


def _iterative_phases(flow: Dict[str, Any], engine: Optional[str] = None, served: bool = False) -> List[set]:
    # Both engines' iterative handlers (core/flow/iterative_process.py, triton/flow/iterative_process.py)
    # force-unload each pre_loop component after it runs, and the loop's and pre_loop's before post_loop,
    # even in eager mode: each pre_loop component is alone, the loop's components are together, the
    # post_loop's are together. The post_loop's are unloaded when the request ends, so the phases hold
    # for every request of a session.
    pre = list(flow.get("pre_loop") or [])
    loop = list((flow.get("loop") or {}).get("components") or [])
    post = list(flow.get("post_loop") or [])
    return [{c} for c in pre] + ([set(loop)] if loop else []) + ([set(post)] if post else [])


def _autoregressive_phases(flow: Dict[str, Any], engine: Optional[str] = None,
                           served: bool = False) -> Optional[List[set]]:
    # An IMAGE decode (`generation.type: autoregressive_image`), read in both handlers:
    #   * the language model, its head, its token embedding and its aligner are loaded for the whole
    #     decode (`_create_session`, `_create_strategy`), with the KV cache;
    #   * triton/flow/autoregressive.py `execute` then releases the language model's weights
    #     (`session.cleanup()` -> `executor.cleanup()`) BEFORE `process_output` loads the decoder — two
    #     phases, on the FIRST request of a process. The handler never unloads the decoder, the head,
    #     the embedding or the aligner (`_unload_non_lm_weights` is not called there), so from the
    #     second request of a session they are loaded during the decode: a SERVED plan has one phase;
    #   * core/flow/autoregressive.py `execute` runs `process_output` (which loads the decoder) with
    #     the language model and its cache still loaded, and unloads after — one phase.
    # THE CACHE IS IN NO PHASE: `session.cleanup()` resets it and keeps its buffers
    # (triton/kv_cache.py `clear`: dropping them would invalidate every recorded replay plan), memoised
    # on the executor. Whoever prices a phase adds the cache to it, whether or not the phase holds the
    # language model.
    # An engine that is not named is held to the wider (compiled) set. A component none of these keys
    # names (an understanding tower the generation never calls) is in no phase: Prism holds it
    # concurrent with every other, nothing says it is unloaded.
    # A TEXT decode declares no phases: its head, and a codec stage where the flow has one, run beside
    # the loaded model in both engines (`_run_snac_codec_decoder` precedes `session.cleanup()`).
    gen = flow.get("generation") or {}
    if gen.get("type") != "autoregressive_image":
        return None
    lm, decoder = gen.get("lm_component"), gen.get("decoder_component")
    if not lm or not decoder:
        return None   # the handlers then fall on their own names; nothing is declared to read
    beside = {c for c in (gen.get(k) for k in ("head_component", "embed_component", "aligner_component")) if c}
    if engine == "triton" and not served:
        return [{lm} | beside, beside | {decoder}]
    return [{lm, decoder} | beside]


def _vlm_phases(flow: Dict[str, Any], engine: Optional[str] = None, served: bool = False) -> Optional[List[set]]:
    # Both engines' VLM handlers (core/flow/vlm.py, triton/flow/vlm.py), in each of their three paths
    # (the legacy splice, the staged splice, the M-RoPE masked splice), run every modality tower and
    # every projection ONCE, before the language model, and unload it right after
    # (`_unload_component_weights`; the compiled engine then `release_flow_memory`) unless the
    # session is persistent — which
    # only an EAGER served plan is (serving/engine.py `_warm_serving`), and an eager plan reads no
    # phases (Prism keeps the sum where nothing loads on demand). Each tower and projection is
    # therefore alone. The language model and its head decode together (register 104: their pair is
    # what the decode holds). The generative-speech leg loads its talker groups beside the
    # still-loaded model (the CFM leg unloads the model after it) or right after releasing it (the
    # deepstack leg): one phase holds the model, its head and every speech component — the bound for
    # both legs. A request that runs no leg never loads a speech component (`_vlm_unloaded`), and
    # Prism prices none of them for it. NEITHER ENGINE EVER UNLOADS THE HEAD OR A SPEECH
    # COMPONENT: `_compute_logits` loads the head and nothing releases it, and the speech legs
    # (speech.py, speech_cfm.py, both engines) unload only the model. A single run ends there; a
    # SERVED session that loads on demand is not persistent, so from its second request every tower
    # and projection loads beside the head and the speech components left loaded by the request
    # before — a served plan's tower phases hold them too. A component no key names is in no phase:
    # Prism holds it concurrent with every other.
    vlm = flow.get("vlm") or {}
    lm = vlm.get("lm_component")
    if not lm:
        return None   # the handler refuses such a flow; nothing is declared to read
    towers = [vlm.get(k) for k in ("vision_component", "vision_projection_component",
                                   "audio_component", "audio_projection_component")]
    speech = {c for c in ((flow.get("speech") or {}).get("components") or {}).values() if isinstance(c, str)}
    head = vlm.get("head_component")
    decode = {lm} | ({head} if head else set()) | speech
    left = (decode - {lm}) if served else set()   # what a request leaves loaded for the next
    return [{t} | left for t in towers if t and t not in decode] + [decode]


#: The request mode whose output is speech (the multimodal family's `modes.supported`): the only mode
#: on which the VLM handlers of both engines run their generative-speech leg — `requests_speech`, read
#: by the handlers' gates and by Prism alike, one statement of the rule.
SPEECH_MODE = "audio"


def requests_speech(mode: Optional[str]) -> bool:
    """True when a request of output `mode` runs a declared generative-speech leg."""
    return str(mode or "") == SPEECH_MODE


def _vlm_unloaded(flow: Dict[str, Any], mode: Optional[str], served: bool) -> set:
    # The speech components (`flow.speech.components`) are loaded by the speech legs alone
    # (core/flow/speech.py, speech_cfm.py and their triton mirrors), and both engines' VLM handlers
    # run a leg only on a request that `requests_speech`; weights load on demand, when a component is
    # run (`ensure_weights_fn`). A single request of another mode never loads them. A SERVED session
    # may take a speech request later, and a request whose mode is not known may be one: both hold
    # the leg (the bound of `_vlm_phases`).
    if served or mode is None or requests_speech(mode):
        return set()
    return {c for c in ((flow.get("speech") or {}).get("components") or {}).values() if isinstance(c, str)}


#: flow type -> the components a request of a given mode NEVER loads, for a single request or a
#: served session. Prism prices none of them — not in a phase, not beside a streamed component, not in
#: a sum. A flow type not here declares nothing: every component may be loaded.
UNLOADED_BY_REQUEST = {
    "vlm": _vlm_unloaded,
}


def unloaded_by_request(topology: Dict[str, Any], mode: Optional[str] = None, served: bool = False) -> set:
    """The components a request of output `mode` never loads (`UNLOADED_BY_REQUEST`); empty when
    its flow type declares none, when the mode is not known, or for a served session."""
    flow = (topology or {}).get("flow") or {}
    rule = UNLOADED_BY_REQUEST.get(flow.get("type"))
    return set(rule(flow, mode, served)) if rule else set()


#: flow type -> the sets of components its handlers hold loaded AT THE SAME TIME, from the topology's
#: flow, for an engine ("compiled" or "triton"; the two handlers of a flow need not release at the same
#: point) and for a single request or a served session (what a handler leaves loaded when a request
#: ends is loaded during the next). Torch-free: Prism reads it to know what sits beside a component it
#: streams and what a plan that loads on demand holds at once. A flow type not here — or a flow whose
#: function returns None — declares no phases, and any of its components may be co-resident with any
#: other.
RESIDENT_PHASES = {
    "iterative_process": _iterative_phases,
    "autoregressive_generation": _autoregressive_phases,
    "vlm": _vlm_phases,
}


def resident_together(topology: Dict[str, Any], engine: Optional[str] = None, served: bool = False):
    """The components a flow holds loaded together on `engine`, as a list of sets, or None when its
    type declares no phases. `served`: the plan is a session's (several requests in one process)."""
    flow = (topology or {}).get("flow") or {}
    phases = RESIDENT_PHASES.get(flow.get("type"))
    return phases(flow, engine, served) if phases else None


def _autoregressive_saved(flow: Dict[str, Any]) -> Optional[set]:
    # An image decode saves what its decoder returns (`process_output`, both engines); the language
    # model's own graph output (its logits over the whole sequence) is an activation, never saved.
    gen = flow.get("generation") or {}
    if gen.get("type") == "autoregressive_image" and gen.get("decoder_component"):
        return {gen["decoder_component"]}
    return None


#: flow type -> the components whose graph output the flow hands to the output boundary (the tensor
#: the run saves). Prism prices the boundary's host bytes from the largest output among them; a flow
#: type not here declares nothing and the largest output of ANY component bounds it.
SAVED_OUTPUT_COMPONENTS = {
    "autoregressive_generation": _autoregressive_saved,
}


def saved_output_components(topology: Dict[str, Any]) -> Optional[set]:
    """The components whose output the flow saves, or None when its type does not declare them."""
    flow = (topology or {}).get("flow") or {}
    saved = SAVED_OUTPUT_COMPONENTS.get(flow.get("type"))
    return saved(flow) if saved else None


def register_flow(flow_type: str):
    """
    Decorator to register flow handler classes.

    Usage:
        @register_flow("iterative_process")
        class IterativeProcessHandler(FlowHandler):
            ...
    """
    def decorator(cls):
        FLOW_REGISTRY[flow_type] = cls
        return cls
    return decorator


def get_flow_handler(flow_type: str, ctx: FlowContext) -> FlowHandler:
    """
    Factory function to create flow handler by type.

    Args:
        flow_type: Flow type string
        ctx: FlowContext for handler

    Returns:
        Instantiated FlowHandler

    Raises:
        RuntimeError: If flow_type not registered (ZERO FALLBACK)
    """
    if flow_type not in FLOW_REGISTRY and flow_type in COMPILED_FLOW_MODULES:
        # The handler registers itself at import; imported here, on the
        # compiled branch's request, never at package import (R33).
        import importlib
        importlib.import_module(COMPILED_FLOW_MODULES[flow_type])
    if flow_type not in FLOW_REGISTRY:
        available = sorted(set(FLOW_REGISTRY) | set(COMPILED_FLOW_MODULES))
        raise RuntimeError(
            f"ZERO FALLBACK: Unknown flow type '{flow_type}'.\n"
            f"Available: {available}"
        )

    handler_class = FLOW_REGISTRY[flow_type]
    return handler_class(ctx)
