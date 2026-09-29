"""The symbol bindings a request's FLOW implies for one component — facts no symbol name carries.

`ActivationProfiler.build_symbol_map` binds a graph's symbols by NAME from one request config
(batch, height, width, time, seq_len). Four flow facts are per component, and a name cannot say them:

* **CFG batch** — the CFG engine runs the flow's LOOP components on [uncond, cond] in one batch
  (`CFGEngine._execute_batched_cfg`), and nothing else; a denoiser that takes a `guidance` input
  embeds the scale and runs no batch-2 pass (`cfg.engine.guidance_embedding_component`).
* **A diffusion encoder's length** — the prompt is tokenized to the encoder's declared length
  (`text.processor.diffusion_max_length`): Open-Sora's T5 runs at 512 tokens, traced at 31.
* **The denoiser's text axis** — a pre-loop encoder's hidden state is FINALIZED before the loop
  (`text_encoder_handler.finalized_text_length`: Wan pads to 512, Sana slices to 300).
* **A FLUX denoiser's packed inputs** — the flow packs the latent (`packed_4d_shape` /
  `packed_5d_shape`) and synthesizes its positional ids and cond (`conditioning_shapes`); its image
  tokens are that packing, not the trace's.
* **A VACE control encoder's pair** — under the loop component's `vace_control_conditioning`
  flag the run feeds the component `global.image` reaches the (inactive, reactive) clips stacked on
  its batch (`image_dsp.VACE_CONTROL_CLIPS`, the CLI's `vace_control_pair_np`).

Prism prices each component with these (the plan), and the derived census keys with the same map:
one binding for both. Each rule CALLS the function the flow itself runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class _Shape:
    """A shape the runtime's binder reads (`SymbolResolver.bind_from_inputs` reads `.shape`)."""

    def __init__(self, shape):
        self.shape = tuple(int(d) for d in shape)


def bind_from_shapes(dag: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, int]:
    """{symbol: value} of a graph fed `inputs` {input name: shape} — the runtime's own binder."""
    from neurobrix.triton.symbols import SymbolResolver
    res = SymbolResolver(dag.get("symbolic_context") or {})
    feed = {f"input::{k}": _Shape(v) for k, v in inputs.items()}
    res.bind_from_inputs(feed, list(feed), dag.get("tensors") or {})
    return dict(res.bindings)


class FlowBindings:
    """A request's flow, as far as symbol bindings go: built once per request from the container's
    topology (`run.request_input_config`), asked per component graph by `build_symbol_map`."""

    def __init__(self, topology: Dict[str, Any], cache_path, container_name: Optional[str] = None,
                 tp_components=()):
        self.topology = topology or {}
        self.cache_path = Path(cache_path) if cache_path is not None else None
        # the name the container registers its flags under: its MANIFEST model_name, never the
        # name the request used (a path, an alias)
        self.container_name = container_name
        self.tp_components = set(tp_components or ())
        self.flow = self.topology.get("flow") or {}
        self._graphs: Dict[str, Dict[str, Any]] = {}

    # -- the container's own graphs ------------------------------------------------------------
    def graph(self, comp: str) -> Dict[str, Any]:
        if comp not in self._graphs:
            self._graphs[comp] = json.loads(
                (self.cache_path / "components" / comp / "graph.json").read_text())
        return self._graphs[comp]

    # -- the rules ------------------------------------------------------------------------------
    def encoder_lengths(self) -> Dict[str, int]:
        """{pre-loop text encoder: tokenized length} of an iterative (diffusion) flow."""
        if self.flow.get("type") != "iterative_process":
            return {}
        from neurobrix.core.module.text.processor import diffusion_max_length
        out = {}
        for enc in self.flow.get("pre_loop") or []:
            tok = "tokenizer" + (("_" + enc.split("text_encoder_", 1)[1])
                                 if enc.startswith("text_encoder_") else "")
            n = diffusion_max_length(self.topology, enc,
                                     (self.topology.get("extracted_values") or {}).get(tok) or {})
            if n:
                out[enc] = n
        return out

    def text_axes(self, input_config) -> Dict[tuple, int]:
        """{(loop component, input name): finalized length} — each pre-loop encoder's HIDDEN STATE
        feeding a loop input, finalized as the handler finalizes it (a pooled vector is no
        sequence)."""
        from neurobrix.core.components.handlers.text_encoder_handler import finalized_text_length
        from neurobrix.core.runtime.registry_flags import get_component_flag
        loop = set((self.flow.get("loop") or {}).get("components") or [])
        pre = self.flow.get("pre_loop") or []
        enc_len = self.encoder_lengths()
        out = {}
        for conn in self.topology.get("connections") or []:
            src, dst = conn.get("from", ""), conn.get("to", "")
            enc, _, oname = src.partition(".")
            comp, _, inp = dst.partition(".")
            if enc not in pre or comp not in loop or "hidden_state" not in oname:
                continue
            n = enc_len.get(enc)
            if n is None:
                from neurobrix.core.prism.profiler import ActivationProfiler
                ge = self.graph(enc)
                be = ActivationProfiler(ge).build_symbol_map(input_config, placement_floor=False,
                                                             flow=False)
                lens = [int(be[s]) for s, i in ((ge.get("symbolic_context") or {}).get("symbols")
                                               or {}).items()
                        if i.get("name") in ("seq_len", "sequence_length") and be.get(s) is not None]
                if not lens:
                    continue
                n = max(lens)
            cfg = dict((self.topology.get("extracted_values") or {}).get("tokenizer") or {})
            if get_component_flag(self.container_name, enc, "zero_pad_embeddings", default=False):
                cfg["zero_pad_embeddings"] = True
            out[(comp, inp)] = finalized_text_length(cfg, n)
        return out

    def packed_denoiser_inputs(self, comp: str, dag: Dict[str, Any], input_config,
                               text_axes: Dict[tuple, int]) -> Optional[Dict[str, list]]:
        """A FLUX denoiser's inputs as the Triton flow builds them (batch 1; the CFG batch after):
        the post-loop VAE's own input at the request, packed, with its ids, cond and text. None for
        any other component."""
        loop = self.flow.get("loop") or {}
        if comp not in (loop.get("components") or []):
            return None
        from neurobrix.triton.flow.iterative_process import TritonIterativeProcessHandler as H
        ins = {sp.get("input_name"): sp for sp in dag["tensors"].values() if sp.get("input_name")}
        four_d = H.declared_packing(self.topology, loop.get("components") or []) is not None
        if "img_ids" not in ins and not four_d:
            return None
        vae = next(iter(self.flow.get("post_loop") or []), None)
        if vae is None:
            return None
        from neurobrix.core.prism.profiler import ActivationProfiler
        from neurobrix.triton.symbols import SymbolResolver
        gv = self.graph(vae)
        bv = ActivationProfiler(gv).build_symbol_map(input_config, placement_floor=False, flow=False)
        rv = SymbolResolver(gv.get("symbolic_context") or {})
        for sid, v in bv.items():
            if v is not None:
                rv._bind(sid, int(v))
        zin = next(sp for sp in gv["tensors"].values() if sp.get("input_name"))
        ss = zin.get("symbolic_shape")
        latent = ([rv.resolve(d) for d in ss["dims"]] if isinstance(ss, dict) and ss.get("dims")
                  else list(zin["shape"]))
        latent = [1, *latent[1:]]
        out: Dict[str, list] = {}
        if len(latent) == 5 and "img_ids" in ins:
            from neurobrix.triton import flux_video_conditioning as FV
            packed = H.packed_5d_shape(latent)
            txt_seq = next((n for (lc, inp), n in text_axes.items() if lc == comp and inp == "txt"),
                           None)
            if txt_seq is None:
                return None
            cs = FV.conditioning_shapes(1, packed[1], packed[2], latent[1], latent[2], latent[3],
                                        latent[4], txt_seq)
            out = {"img": packed, "img_ids": cs["img_ids"], "txt_ids": cs["txt_ids"],
                   "cond": cs["cond"], "txt": [1, txt_seq, ins["txt"]["shape"][-1]]}
        elif len(latent) == 4 and four_d:
            out[loop.get("state_input", "hidden_states")] = H.packed_4d_shape(latent)
            for (lc, inp), n in text_axes.items():
                if lc == comp and inp in ins:
                    out[inp] = [1, n, ins[inp]["shape"][-1]]
        else:
            return None
        for name, sp in ins.items():
            if name not in out:
                out[name] = [1, *sp["shape"][1:]] if sp.get("shape") else [1]
        return out

    def vace_control_batch(self, comp: str) -> Optional[int]:
        """The batch the run feeds `comp` when it is the VACE control encoder: the component the
        image input reaches, under a loop component carrying `vace_control_conditioning` (the flag
        the CLI builds the pair on). None for any other component."""
        from neurobrix.core.runtime.registry_flags import get_component_flag
        from neurobrix.core.module.vision.image_dsp import VACE_CONTROL_CLIPS
        loop = (self.flow.get("loop") or {}).get("components") or []
        if not any(get_component_flag(self.container_name, c, "vace_control_conditioning", default=None)
                   for c in loop):
            return None
        fed = {conn.get("to", "").partition(".")[0] for conn in self.topology.get("connections") or []
               if conn.get("from") == "global.image"}
        return len(VACE_CONTROL_CLIPS) if comp in fed else None

    def overrides(self, dag: Dict[str, Any], input_config) -> Dict[str, int]:
        """{symbol id: value} this component's flow imposes on top of the name-driven map."""
        comp = dag.get("component_name")
        if not comp:
            return {}
        table = (dag.get("symbolic_context") or {}).get("symbols") or {}
        out: Dict[str, int] = {}
        pair = self.vace_control_batch(comp)
        if pair:
            for sid, info in table.items():
                if info.get("name") == "batch":
                    out[sid] = pair
        enc_len = self.encoder_lengths()
        if comp in enc_len:
            for sid, info in table.items():
                if info.get("name") in ("seq_len", "sequence_length"):
                    out[sid] = enc_len[comp]
        loop = set((self.flow.get("loop") or {}).get("components") or [])
        if comp not in loop:
            return out
        axes = self.text_axes(input_config)
        packed = self.packed_denoiser_inputs(comp, dag, input_config, axes)
        if packed is not None:
            out.update(bind_from_shapes(dag, packed))
        else:
            for (lc, inp), n in axes.items():
                if lc != comp:
                    continue
                for sid, info in table.items():
                    src = info.get("source") or ""
                    if src == f"input::{inp}::dim_1" or ("mask" in src and src.endswith("::dim_1")):
                        out[sid] = n
        from neurobrix.triton.cfg.engine import guidance_embedding_component
        if (comp not in self.tp_components and guidance_embedding_component(self.topology) is None
                and input_config.batch_size):
            for sid, info in table.items():
                if info.get("name") == "batch":
                    out[sid] = int(input_config.batch_size)
        return out
