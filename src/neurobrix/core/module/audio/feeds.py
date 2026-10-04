"""What an audio flow feeds its first stage for one recording — computed without running it.

The flows feed the recording's OWN extent (never the trace's, `core/flow/input_extent`), so the
extent a component runs at is a fact of the request: Prism prices the largest feed
(`core/prism/flow_bindings.FlowBindings.audio_feeds`) and the derived census keys every feed
(`tools/derived_census.py`) from this ONE function. It calls what the flows call: the shared
loader and numpy extractors (`mel_dsp` — the Triton flows' own; the compiled rnnt flow computes
its mel in torch from the same loader's samples, so the frame count is the same number), the RNNT
feed plan (`stt_longform.rnnt_feed_plan`), the admission door (`input_extent.admit`). Numpy +
stdlib only: both engines' planning paths import it (R33).

A container that cannot take the recording's extent is refused BY NAME here, at plan time, before
a weight is loaded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

#: The dither of the NeMo extractor draws noise; a shape-only pass draws from its own generator so
#: that planning never advances the stream the run's own features draw from.
_SHAPE_ONLY_SEED = 0


def first_audio_stage(topology: Optional[dict]) -> Optional[str]:
    """The component an audio flow feeds the recording's features to — by the flows' own rule:
    the rnnt flow always takes a recording; every other audio flow takes one when its input
    modality is audio, a modality it does not declare being read off its direction (`stt` is
    audio in, `core/flow/audio.py::_preprocess_input`). None for every other flow."""
    flow = (topology or {}).get("flow") or {}
    audio = flow.get("audio") or {}
    stages = audio.get("stages") or []
    if not stages:
        return None
    if flow.get("type") != "rnnt":
        modality = (audio.get("input") or {}).get(
            "modality", "audio" if audio.get("direction") == "stt" else "text")
        if modality != "audio":
            return None
    return stages[0].get("component")


def request_feeds(topology: dict, cache_path, audio_path, dag: dict,
                  container: Optional[str] = None, family: Optional[str] = None
                  ) -> List[Dict[str, tuple]]:
    """Every distinct `{input name: shape}` the flow feeds its first audio stage (`dag`, that
    stage's graph) for the recording at `audio_path`, in feed order — one entry for a recording
    that goes through in one pass, one per distinct window extent of a long-form RNNT run."""
    from neurobrix.core.flow import input_extent as IE
    from neurobrix.core.module.audio import mel_dsp
    comp = first_audio_stage(topology)
    if comp is None:
        return []
    flow = topology.get("flow") or {}
    audio = flow.get("audio") or {}
    cfg = mel_dsp.model_config_dir(Path(cache_path))
    rng = np.random.default_rng(_SHAPE_ONLY_SEED)
    if flow.get("type") == "rnnt":
        from neurobrix.core.config.loader import get_family_config
        from neurobrix.core.module.audio.stt_longform import rnnt_feed_plan
        params = mel_dsp.nemo_mel_params(cfg)
        feats = mel_dsp._nemo_mel(str(audio_path), cfg, None, rng=rng)      # the Triton flow's call
        feat_in, len_in = IE.features_and_length(dag)
        if not family:
            raise RuntimeError(
                f"container {container!r}: the RNNT feed plan reads the family profile's long-form "
                f"window, and no family was given.")
        plan, _overlap = rnnt_feed_plan(feats.shape[2], get_family_config(family).get("long_form"),
                                        params["sr"], params["hop"], family)
        feeds: List[Dict[str, tuple]] = []
        for _start, valid in plan:
            feed = {feat_in: (1, int(feats.shape[1]), int(valid)), len_in: (1,)}
            if feed not in feeds:
                IE.admit(container, comp, dag, feed)
                feeds.append(feed)
        return feeds
    inp = audio.get("input") or {}
    variable = inp.get("variable", "global.input_features")
    axes = IE.input_axes(dag, IE.fed_input(topology, comp, variable))
    if axes is None:
        raise IE.FrozenTraceExtent(
            f"container {container!r}, component {comp!r}: the graph has no input for the flow's "
            f"{variable!r}.")
    prep = mel_dsp.resolve_preprocessing(inp.get("preprocessing"), axes.trace_shape)
    feats = mel_dsp.extract_features_np(prep, str(audio_path), cfg, axes.trace_shape, rng=rng)
    feed = {axes.name: tuple(int(d) for d in feats.shape)}
    IE.admit(container, comp, dag, feed)
    return [feed]
