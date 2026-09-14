# The `nbx_tensor` boundary — a library the engine imports, never the reverse

Stated by the owner on 2026-09-14, for the trunk: `nbx_tensor`
(`src/neurobrix/kernels/nbx_tensor.py`, the `NBXTensor` and its
`DeviceAllocator`) is the replacement of torch on the Triton branch — R33 has
said so from the start — and may one day be distributed as a separate library
under Apache 2.0 that the engine imports. Not in this loop and not this month:
**the criterion for separating it is one month without an API change forced by
the engine**, and this week is not that. But its boundary stays clean from now:

* no import of the engine leaks into `nbx_tensor` — not conditional, not
  inside a function, not for a "small" convenience;
* it has its own test suite (`tests/unit/nbx_tensor/`) and its own page (this
  one);
* every function added to it is added the way a library gains one: named,
  documented, tested alone;
* when something is missing on the Triton path, it is GIVEN to `nbx_tensor`
  rather than fetched from torch or worked around;
* the separation, when it comes, is a `git subtree`, never a rewrite.

## The gate

`tests/unit/nbx_tensor/test_the_boundary_does_not_widen.py` records every
import of `neurobrix.*` or `torch` the file carried on 2026-09-14, by the scope
that carries it, and fails on any addition with the rule in its message. The
record can only shrink: a recorded import that disappears must leave the
record too, so the ratchet tightens. Seen red on an injected engine import
before it was trusted.

## The census on 2026-09-14 — what separation still has to undo

Read by AST (`boundary_imports()`), 18 sites in three classes. None is new
and none may gain a sibling.

| class | sites | what they are | what the separation does with them |
|---|---:|---|---|
| `torch` | 3 | `nbx_dtype_to_torch`, `set_torch_device`, `nbx_to_torch` — the compiled-mode boundary conversions | an optional interop adapter OUTSIDE the library (the engine's compiled branch owns it); the library never imports torch |
| `neurobrix.kernels.wrappers` | 9 | the operator overloads `__add__ … __neg__` dispatch to the engine's elementwise wrappers; `DeviceAllocator._pool_parked_cap` reads the hardware profile | invert the dependency: the library exposes a registry the engine fills at import (operators), and the allocator receives its cap as a parameter |
| `neurobrix.kernels.ops.*` | 6 | `strided_copy`, `fill_op`, `copy_op`, `cat_op` — the library's own Triton kernels (contiguous, fill, device copy, cat) | they travel WITH the library in the subtree; they are its kernels, not the engine's |

Sibling relative imports (`.metal_device`) and the library's own dependencies
(`triton`, `numpy`) are not boundary crossings.

## Where the suite lives

`tests/unit/nbx_tensor/` — the tests that exercise the library alone (host
reads of any size, the boundary ratchet). Tests under `tests/unit/kernels/`
that use `NBXTensor` as a substrate to test an engine kernel stay where they
are: they test the kernel, not the library.
