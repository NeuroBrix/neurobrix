"""The PyTorch engine pins its host-held weights in ONE registered block, in place (core/strategies/pinned_block.py).

Per-weight `pin_memory()` made a pinned copy of every weight through PyTorch's host allocator, which rounds each block up
to a power of two (a 3 MB expert takes 4 MB, 2.6 GiB takes 4 GiB, measured 2026-11-06), while the originals stayed alive:
Qwen3-Coder-30B's native expert store held 60.5 GB of originals beside 76.5 GB of pinned copies for 57.6 GB of weights.
One block registered at its exact size: steady 61 GB, the same output. What this test does if the code were wrong: a
view that is not pinned, a value that changed, a block larger than the weights' bytes (rounded), or a block left
registered once its holder goes — red. Needs a card (the driver registers host memory)."""
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a card")


def test_the_weights_become_pinned_views_of_one_block_of_their_exact_size():
    from neurobrix.core.strategies.pinned_block import pin_in_one_block, _ALIGN
    g = torch.Generator().manual_seed(0)
    w = {"a": torch.randn(3, 1000, generator=g), "b": torch.randn(77, generator=g).half(),
         "c": torch.randn(64, 32, generator=g).t()}                      # a non-contiguous one
    ref = {k: v.clone() for k, v in w.items()}
    holder = pin_in_one_block(w)
    assert holder is not None
    want = sum(-(-v.numel() * v.element_size() // _ALIGN) * _ALIGN for v in ref.values())
    assert holder.nbytes == want
    for k in w:
        assert w[k].is_pinned() and torch.equal(w[k], ref[k]) and w[k].dtype == ref[k].dtype
        assert w[k].untyped_storage().data_ptr() == holder.block.untyped_storage().data_ptr()
    assert pin_in_one_block(w) is None                                   # already pinned: nothing to do


def test_the_block_is_unregistered_when_its_holder_goes(monkeypatch):
    from neurobrix.core.strategies import pinned_block as PB
    w = {"a": torch.ones(1 << 16)}
    holder = PB.pin_in_one_block(w)
    ptr = holder.block.data_ptr()
    real = torch.cuda.cudart()
    seen = []

    class _Rt:
        def __getattr__(self, n):
            return getattr(real, n)

        def cudaHostUnregister(self, p):
            seen.append(p)
            return real.cudaHostUnregister(p)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: _Rt())
    del holder
    assert seen == [ptr]
