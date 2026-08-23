"""Triton Arena — NBXTensor slot storage for compiled execution.

Memory layout: [weights | inputs | intermediates]
All slots hold NBXTensor. Zero torch dependency.
"""

class Arena:
    """O(1) tensor storage via integer-indexed list."""

    __slots__ = ('_slots', '_num_weights', '_num_inputs')

    def __init__(self, size: int, num_weights: int = 0, num_inputs: int = 0):
        self._slots: list = [None] * size
        self._num_weights = num_weights
        self._num_inputs = num_inputs

    def __getitem__(self, idx: int):
        return self._slots[idx]

    def __setitem__(self, idx: int, value):
        self._slots[idx] = value

    def __len__(self) -> int:
        return len(self._slots)

    def restore_from(self, snapshot: list) -> None:
        """Bulk-restore every slot from a same-length snapshot list.

        C-level slice assign. The frozen-plan replay restores the full
        arena after every replayed step; the per-slot Python loop it
        replaces cost ~140k `__setitem__` calls per decode token on the
        30B row (2026-08-23 host profile), most of one of the two
        ~16 ms GPU-idle holes in the step.
        """
        if len(snapshot) != len(self._slots):
            raise ValueError(
                f"arena restore length mismatch: snapshot "
                f"{len(snapshot)} vs arena {len(self._slots)}")
        self._slots[:] = snapshot

    def clear_intermediates(self):
        """Free intermediate slots (keep weights + inputs)."""
        start = self._num_weights + self._num_inputs
        for i in range(start, len(self._slots)):
            self._slots[i] = None

    def clear_inputs(self) -> None:
        """Clear input tensors (for next inference)."""
        start = self._num_weights
        end = self._num_weights + self._num_inputs
        for i in range(start, end):
            self._slots[i] = None

    def clear_all(self) -> None:
        """Clear all tensors."""
        for i in range(len(self._slots)):
            self._slots[i] = None
