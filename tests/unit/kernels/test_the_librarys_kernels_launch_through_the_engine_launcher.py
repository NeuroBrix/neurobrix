"""A kernel the engine moved into its tensor library (nbxtensor.ops) is still launched by the engine's launcher
(`kernels/launcher._is_ours`): its certified-only check, its census shadow and its key recording see it. M1 zone 3
(2026-10-29): the launcher owned only `neurobrix.*`, so the moved kernels went to upstream Triton's launch — on a card
the outputs were identical and nothing said so; without a card every census shadow failed ("Backend device 'nvidia'
is not active"). No card."""
import pkgutil
import importlib

import nbxtensor.ops as lib_ops
from triton.runtime.jit import JITFunction

from neurobrix.kernels.launcher import _is_ours


def test_every_jit_kernel_of_the_library_is_the_launchers():
    seen = 0
    for mi in pkgutil.iter_modules(lib_ops.__path__):
        mod = importlib.import_module(f"{lib_ops.__name__}.{mi.name}")
        for obj in vars(mod).values():
            if isinstance(obj, JITFunction) and obj.fn.__module__ == mod.__name__:
                assert _is_ours(obj), f"{mod.__name__}.{obj.fn.__name__} is launched outside the engine's launcher"
                seen += 1
    assert seen > 100, seen
