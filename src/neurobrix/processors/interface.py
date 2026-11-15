"""The call interfaces by `kind`, what the engine lends a processor (`Ask`), and the processor as the engine holds it.

A processor is standard-library text work: it never holds a tensor, a device or an array (anything tensor is
nbxtensor's). Today one kind is spoken, `text_front_end`:

    count(text, variant) -> int            tokens, before any network runs
    units(text, variant, ask) -> [str]     the unit sequence the model's own config maps to ids

`ask` is all the engine lends: `ask.units(role, text, variant)` calls another declared processor (the record's
`uses`), `ask.labels(role, rows)` runs a declared component of the model on the engine's executor and answers plain
integers (the record's `components`; the engine supplies the runner).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from neurobrix.processors import messages
from neurobrix.processors.declaration import Declaration

#: kind -> the highest `interface` number this engine speaks for it
INTERFACES: Dict[str, int] = {"text_front_end": 1}
#: kind -> the methods a processor of that kind defines
METHODS: Dict[str, tuple] = {"text_front_end": ("count", "units")}


def refuse_unspoken(decl: Declaration) -> None:
    """A kind this engine does not know, or an interface newer than the one it speaks, is refused before a byte moves."""
    if decl.interface > INTERFACES.get(decl.kind, 0):
        raise messages.refusal(messages.TOO_NEW, label=decl.label)


def refuse_missing_components(decl: Declaration, components) -> None:
    """Every component the declaration's `components` names is a part of this container."""
    for component in decl.components.values():
        if component not in components:
            raise messages.refusal(messages.NO_COMPONENT, label=decl.label, component=component)


class ArtifactData:
    """The `data/` tree of a verified artifact, read from its bytes in memory: what a processor is built with."""

    def __init__(self, archive, prefix: str = "data/"):
        self._archive = archive
        self._prefix = prefix

    def names(self) -> List[str]:
        """Every file of the tree, by its path relative to the tree."""
        return [n[len(self._prefix):] for n in self._archive.namelist()
                if n.startswith(self._prefix) and not n.endswith("/")]

    def read(self, path: str) -> bytes:
        """The bytes of one file of the tree."""
        return self._archive.read(self._prefix + path)

    def text(self, path: str) -> str:
        """One file of the tree as UTF-8 text."""
        return self.read(path).decode("utf-8")


class Ask:
    """What the engine lends one processor: the processors it `uses`, and the runner of the components it names."""

    def __init__(self, decl: Declaration, used: Dict[str, "Held"], labels: Optional[Callable]):
        self._decl = decl
        self._used = used
        self._labels = labels

    def units(self, role: str, text: str, variant: str) -> List[str]:
        """The units the processor this one uses as `role` gives for `text`."""
        return self._used[role].units(text, variant)

    def labels(self, role: str, rows) -> List[int]:
        """The labels the model's component this processor names as `role` gives for `rows`."""
        component = self._decl.components[role]
        if self._labels is None:
            raise messages.refusal(messages.NOT_LENT, label=self._decl.label, component=component)
        return [int(v) for v in self._labels(component, rows)]


class Held:
    """A loaded processor bound to its declaration: the variant a request needs is one the declaration lists."""

    def __init__(self, decl: Declaration, processor, ask: Ask):
        for method in METHODS[decl.kind]:
            if not callable(getattr(processor, method, None)):
                raise messages.refusal(messages.TOO_NEW, label=decl.label)
        self.declaration = decl
        self._processor = processor
        self._ask = ask

    def _variant(self, variant: str) -> str:
        if variant not in self.declaration.variants:
            raise messages.refusal(messages.VARIANT, label=self.declaration.label, variant=variant,
                                   variants=", ".join(self.declaration.variants))
        return variant

    def count(self, text: str, variant: str) -> int:
        """Tokens `text` becomes, before any network runs."""
        return int(self._processor.count(text, self._variant(variant)))

    def units(self, text: str, variant: str) -> List[str]:
        """The unit sequence of `text`, which the model's own config maps to ids."""
        return list(self._processor.units(text, self._variant(variant), self._ask))
