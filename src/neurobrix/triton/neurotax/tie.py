"""The tie reader: what a container declares about its token tables, its tied weights, its hidden output and the
source of each stage's logits, read from topology.json and returned frozen. Names only: nothing is loaded or bound.

THE DECLARATIONS (the tie rulings D1-D7). topology.json declares, in NeuroTax names:
  components.<c>.token_embedding   one weight key of c (the token table c holds);
  components.<c>.tied_weights      [{"param", "source_component", "source_weight"}]: c's weight `param` IS the
                                   tensor `source_weight` of `source_component`, stored once, there;
  components.<c>.hidden_output     a tensor id of c's graph: the tensor a head graph or the tied projection reads;
  <stage>.logits_source            one of three words, on flow.stages[i], on flow.<section>.stages[i] (the
                                   sections of today: audio, vlm) and on flow.generation: "head" (a head graph of the
                                   container is run), "self" (the graph's own output is the logits), "tied" (the
                                   projection on the declared tie, for containers whose graph is not rebuilt yet).
A "head" stage names its head by `head_component`: on the stage, else on its section, else on flow.generation.

TWO FORMS (carries_the_new_keys, the one line between them). A topology none of whose components carries one of
NEW_KEYS as a direct key is of the OLD FORM: it reads as not declared (old_form=True), its words are left to today's
path, never mapped and never refused here. A topology one of whose components carries one is of the tie form, read
strictly as below. A new key anywhere else is MISPLACED and refused by name, in either form.

THE DOOR. A key that is absent is NOT DECLARED (None, or an empty tuple): the reader never guesses one. A key that is
present and malformed is REFUSED BY NAME (TieDeclarationError): the message names the container, the component or
stage, the key, what is wrong, and the remedy. A declaration is never ignored: a logits_source on a stage that names no
component, or anywhere under the flow outside the stages above, is refused too. No default, no warning-and-continue.

This file exists once per branch (pytorch/neurotax, triton/neurotax), byte-identical; standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

#: the three words a stage may declare as the source of its logits
LOGITS_SOURCES = ("head", "self", "tied")
#: the keys of one tied_weights entry
TIE_KEYS = ("param", "source_component", "source_weight")
#: the keys only a container of the tie form carries; a topology with none of them is of the OLD FORM
NEW_KEYS = ("token_embedding", "tied_weights", "hidden_output")


class TieDeclarationError(ValueError):
    """A tie declaration of the container is malformed; the message names it and its remedy."""


@dataclass(frozen=True)
class TiedWeight:
    """One declared tie: the consumer's weight `param` is `source_weight` of `source_component`."""
    param: str
    source_component: str
    source_weight: str


@dataclass(frozen=True)
class ComponentTie:
    """What one component declares: its token table, its ties, its hidden output (None / () = not declared)."""
    component: str
    token_embedding: Optional[str]
    tied_weights: Tuple[TiedWeight, ...]
    hidden_output: Optional[str]

    @property
    def declared(self) -> bool:
        return self.token_embedding is not None or bool(self.tied_weights) or self.hidden_output is not None


@dataclass(frozen=True)
class StageLogits:
    """One stage: where it sits in the topology, its component, its declared logits source and head component."""
    where: str
    component: Optional[str]
    logits_source: Optional[str]
    head_component: Optional[str]


@dataclass(frozen=True)
class TieReading:
    """The whole container's tie declarations, in topology order. `old_form`: the topology carries none of the
    NEW_KEYS; nothing of it is read as a tie declaration (no component declares, no stage is listed) and its stages'
    words stay the old path's."""
    container: str
    components: Tuple[ComponentTie, ...]
    stages: Tuple[StageLogits, ...]
    old_form: bool = False

    @property
    def declared(self) -> bool:
        return any(c.declared for c in self.components) or any(s.logits_source is not None for s in self.stages)

    def component(self, name: str) -> ComponentTie:
        for c in self.components:
            if c.component == name:
                return c
        raise KeyError(f"TIE READING: container '{self.container}' has no component '{name}'")


def _refuse(container: str, place: str, key: str, wrong: str, declared: str) -> TieDeclarationError:
    return TieDeclarationError(
        f"TIE DECLARATION: container '{container}', {place}, key '{key}': {wrong}.\n"
        f"  FIX: rebuild the container with the builder so that {declared} is declared.")


def _name(value, container: str, place: str, key: str, declared: str) -> Optional[str]:
    """An optional non-empty string key: None when absent, refused when present and not a non-empty string."""
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise _refuse(container, place, key, f"{value!r} is not a non-empty name", declared)
    return value


def _tie(entry, index: int, ctx: Tuple[str, str, frozenset]) -> TiedWeight:
    container, comp, names = ctx
    place, key = f"component '{comp}'", f"tied_weights[{index}]"
    declared = "each tie as {param, source_component, source_weight}"
    if not isinstance(entry, dict):
        raise _refuse(container, place, key, f"{entry!r} is not an object", declared)
    for k in TIE_KEYS:
        if not isinstance(entry.get(k), str) or not entry.get(k):
            raise _refuse(container, place, f"{key}.{k}", "missing, or not a non-empty name", declared)
    extra = sorted(set(entry) - set(TIE_KEYS))
    if extra:
        raise _refuse(container, place, key, f"holds keys other than {list(TIE_KEYS)}: {extra}", declared)
    if entry["source_component"] not in names:
        raise _refuse(container, place, f"{key}.source_component",
                      f"'{entry['source_component']}' is not a component of the topology",
                      "the source component as a component of the container")
    if entry["source_component"] == comp and entry["source_weight"] == entry["param"]:
        raise _refuse(container, place, key, f"'{entry['param']}' is tied to itself",
                      "the tie's source as another weight")
    return TiedWeight(entry["param"], entry["source_component"], entry["source_weight"])


def _ties(value, ctx: Tuple[str, str, frozenset]) -> Tuple[TiedWeight, ...]:
    container, comp, _ = ctx
    if value is None:
        return ()
    place = f"component '{comp}'"
    if not isinstance(value, list) or not value:
        raise _refuse(container, place, "tied_weights", f"{value!r} is not a non-empty list",
                      "tied_weights as a list of ties")
    ties = tuple(_tie(e, i, ctx) for i, e in enumerate(value))
    params = [t.param for t in ties]
    dup = sorted({p for p in params if params.count(p) > 1})
    if dup:
        raise _refuse(container, place, "tied_weights", f"the param(s) {dup} are tied more than once",
                      "each tied param once")
    return ties


def _component(container: str, comp: str, spec, names: frozenset) -> ComponentTie:
    if not isinstance(spec, dict):
        return ComponentTie(comp, None, (), None)
    place = f"component '{comp}'"
    return ComponentTie(
        comp,
        _name(spec.get("token_embedding"), container, place, "token_embedding", "token_embedding as one weight key"),
        _ties(spec.get("tied_weights"), (container, comp, names)),
        _name(spec.get("hidden_output"), container, place, "hidden_output", "hidden_output as one tensor id"))


def _stage(container: str, where: str, stage: dict, heads: Tuple[dict, ...], component_key: str) -> StageLogits:
    place = f"stage {where}"
    comp = _name(stage.get(component_key), container, place, component_key, f"the stage's {component_key}")
    source = stage.get("logits_source")
    if source is not None and source not in LOGITS_SOURCES:
        raise _refuse(container, place, "logits_source", f"{source!r} is not one of {list(LOGITS_SOURCES)}",
                      "logits_source as one of " + ", ".join(LOGITS_SOURCES))
    if source is not None and comp is None:
        raise _refuse(container, place, component_key, f"the stage declares logits_source {source!r} and names no "
                      f"component ('{component_key}' absent)", f"the stage's {component_key}")
    head = next((h["head_component"] for h in (stage,) + heads if "head_component" in h), None)
    head = _name(head, container, place, "head_component", "the head component")
    return StageLogits(where, comp, source, head)


def _sections(flow: dict) -> Tuple[Tuple[str, dict], ...]:
    """The places that hold stages: the flow itself, then each section of it that holds a stage list, in order."""
    found = [("flow", flow)]
    found += [(f"flow.{k}", v) for k, v in flow.items() if isinstance(v, dict) and isinstance(v.get("stages"), list)]
    return tuple(found)


def _carriers(node, path: str):
    """Every place under the flow that carries a logits_source key, by its path."""
    if isinstance(node, dict):
        if "logits_source" in node:
            yield path
        for k, v in node.items():
            yield from _carriers(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _carriers(v, f"{path}[{i}]")


def _refuse_strays(container: str, flow, read: Tuple[StageLogits, ...]) -> None:
    """A logits_source the reader does not read as a stage is refused by name, never ignored."""
    stray = sorted(set(_carriers(flow, "flow")) - {s.where for s in read})
    if stray:
        raise _refuse(container, f"flow path {stray[0]}", "logits_source", "declared outside a stage the engine "
                      "reads (flow.stages[i], flow.<section>.stages[i], flow.generation)",
                      "logits_source on the stage it belongs to")


def _stages(container: str, flow) -> Tuple[StageLogits, ...]:
    if not isinstance(flow, dict):
        return ()
    gen = flow.get("generation") if isinstance(flow.get("generation"), dict) else {}
    found = []
    for prefix, section in _sections(flow):
        heads = (section, gen) if section is not flow else (gen,)
        stages = section.get("stages")
        for i, st in enumerate(stages if isinstance(stages, list) else []):
            if isinstance(st, dict):
                found.append(_stage(container, f"{prefix}.stages[{i}]", st, heads, "component"))
    if gen:
        found.append(_stage(container, "flow.generation", gen, (), "lm_component"))
    _refuse_strays(container, flow, tuple(found))
    return tuple(found)


def _check_stage(reading: TieReading, stage: StageLogits) -> None:
    container, place = reading.container, f"stage {stage.where}"
    names = {c.component for c in reading.components}
    if stage.logits_source is None:
        return
    if stage.component not in names:
        raise _refuse(container, place, "component", f"{stage.component!r} is not a component of the topology",
                      "the stage's component")
    comp = reading.component(stage.component)
    if stage.logits_source == "head" and stage.head_component not in names:
        raise _refuse(container, place, "logits_source", f"'head' names no head component of the topology "
                      f"(head_component {stage.head_component!r})", "the head component of a 'head' stage")
    if stage.logits_source == "tied" and not comp.tied_weights:
        raise _refuse(container, place, "logits_source", f"'tied' but component '{comp.component}' declares no "
                      "tied_weights", "the tie of a 'tied' stage (tied_weights)")
    if stage.logits_source == "tied" and comp.hidden_output is None:
        raise _refuse(container, place, "logits_source", f"'tied' but component '{comp.component}' declares no "
                      "hidden_output", "the hidden output the tied projection reads (hidden_output)")


def _check_hidden_is_read(reading: TieReading) -> None:
    """D3: a hidden_output is declared only where it is read: its component is the component of a 'tied' or 'head'
    stage (the tied projection or the head graph reads it); anywhere else it is declared and unused, refused."""
    readers = {s.component for s in reading.stages if s.logits_source in ("tied", "head")}
    for comp in reading.components:
        if comp.hidden_output is not None and comp.component not in readers:
            raise _refuse(reading.container, f"component '{comp.component}'", "hidden_output",
                          f"'{comp.hidden_output}' is declared and nothing reads it (no 'tied' or 'head' stage of "
                          f"component '{comp.component}')",
                          "a hidden_output only where a 'tied' or 'head' stage reads it")


def carries_the_new_keys(topology) -> bool:
    """THE LINE BETWEEN THE TWO FORMS: a topology is of the tie form when one of the NEW_KEYS is a direct key of one
    component's object (components.<c>.<key>). Otherwise it is of the old form: its logits_source words (lm_head,
    embed_weight_tied, self) are the old path's, read by no one here, mapped by no one here, and never refused here.
    A new key anywhere else is not this line's: it is misplaced, refused by name (_refuse_misplaced)."""
    specs = topology.get("components") if isinstance(topology, dict) else None
    if not isinstance(specs, dict):
        return False
    return any(isinstance(v, dict) and any(k in NEW_KEYS for k in v) for v in specs.values())


def _new_key_paths(node, path: str):
    """Every place a NEW_KEYS key appears as a key, by its path."""
    if isinstance(node, dict):
        for k, v in node.items():
            here = f"{path}.{k}" if path else str(k)
            if k in NEW_KEYS:
                yield here
            yield from _new_key_paths(v, here)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _new_key_paths(v, f"{path}[{i}]")


def _refuse_misplaced(container: str, topology: dict) -> None:
    """A new key that is not a direct key of a component's object is MISPLACED, refused by name in either form."""
    specs = topology.get("components") if isinstance(topology.get("components"), dict) else {}
    placed = {f"components.{c}.{k}" for c, v in specs.items() if isinstance(v, dict) for k in v if k in NEW_KEYS}
    misplaced = sorted(set(_new_key_paths(topology, "")) - placed)
    if misplaced:
        key = misplaced[0].rsplit(".", 1)[-1]
        raise _refuse(container, f"topology path {misplaced[0]}", key, "misplaced: a tie key is declared only as a "
                      "direct key of one component (components.<c>." + key + ")", f"{key} on its component")


def read_tie(topology, container: Optional[str] = None) -> TieReading:
    """The container's tie declarations, frozen. A misplaced new key: refused. Old form (no new key on a component):
    not declared, never refused. Tie form: absent keys are not declared, malformed ones refused by name."""
    if not isinstance(topology, dict):
        raise TieDeclarationError(f"TIE DECLARATION: the topology is {type(topology).__name__}, not an object")
    container = container or str(topology.get("model_name") or "?")
    specs = topology.get("components") if isinstance(topology.get("components"), dict) else {}
    _refuse_misplaced(container, topology)
    if not carries_the_new_keys(topology):
        return TieReading(container, tuple(ComponentTie(c, None, (), None) for c in specs), (), old_form=True)
    names = frozenset(specs)
    reading = TieReading(container, tuple(_component(container, c, s, names) for c, s in specs.items()),
                         _stages(container, topology.get("flow")))
    for stage in reading.stages:
        _check_stage(reading, stage)
    _check_hidden_is_read(reading)
    return reading
