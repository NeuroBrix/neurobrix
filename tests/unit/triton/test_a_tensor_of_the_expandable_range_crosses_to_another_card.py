"""A tensor held in a card's expandable range is handed to another card (CARDS REQUIRED: two that are peers).

The device pool grows by mapping chunks into one virtual range (`nbxtensor.memory.vmm`). A mapped chunk is readable
only from the devices its access grant names — enabling peer access between the two contexts does nothing for it
(NVIDIA: forums.developer.nvidia.com/t/148875). Granted to its owner alone, every tensor of the range refused the
card-to-card copy: "cudaMemcpy failed rc=1 (kind=3)". A hand-off of a few hundred kilobytes passed (it lives in an
ordinary block), so a mixture-of-experts decode on two cards ran; a dense model split over two cards died at its
first hand-off (GLM-4.1V-9B-Thinking on two V100-16GB, 271 x 4096 fp32, 2026-11-11 — and on the published 0.6.7).

What this test would do if the code were wrong: the 4 MB and 16 MB hand-offs raise — RED (seen); or arrive with
other values than they left with."""
import numpy as np
import pytest

pytest.importorskip("triton")

from nbxtensor.tensor import NBXTensor  # noqa: E402


@pytest.mark.parametrize("elements", [4096, 1 << 20, 1 << 22])
def test_the_values_arrive_on_the_other_card(elements, needs_device, declared_profile):
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.memory.device_transfer import transfer_tensor
    if DeviceAllocator.device_count() < 2:
        pytest.skip("two cards are needed")
    values = (np.arange(elements, dtype=np.float32) % 8191.0)
    here = NBXTensor.from_numpy(values)
    src = here._device_idx
    dst = next(d for d in range(DeviceAllocator.device_count()) if d != src)
    if not DeviceAllocator.can_access_peer(src, dst):
        pytest.skip(f"cards {src} and {dst} are not peers: the hand-off goes through the host")
    there = transfer_tensor(here, dst)
    assert there._device_idx == dst
    back = transfer_tensor(there, src)                       # and the other direction, from the second card's range
    DeviceAllocator.set_device(src)
    assert np.array_equal(np.asarray(back.numpy()).ravel(), values)
