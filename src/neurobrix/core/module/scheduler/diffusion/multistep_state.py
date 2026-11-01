"""The resumable state of a multistep solver (DPM-Solver++, UniPC): the step index and the solver HISTORY travel with
the latent — the one definition both schedulers inherit."""
import numpy as np


class MultistepStateMixin:
    def checkpoint_state(self):
        """Resumable, but only because the solver HISTORY travels with the
        latent. A multistep solver computes the next sample from the last
        `solver_order` model outputs; restoring the latent alone would
        silently produce a different image from the uninterrupted run.

        Tensors leave as numpy so the state stays serialisable and torch-free
        (the triton engine needs the same checkpoint, and R33 keeps torch out
        of that tree).
        """
        import numpy as np

        def _out(t):
            return None if t is None else np.asarray(t.detach().cpu().numpy())

        state = {
            "_step_index": self._step_index,
            "lower_order_nums": self.lower_order_nums,
            "model_outputs": [_out(t) for t in self.model_outputs],
        }
        if hasattr(self, "this_order"):
            # The order chosen for the CURRENT step. Missing it leaves the
            # resumed solver building an empty coefficient list — the resume
            # crashed on `stack expects a non-empty TensorList` before this
            # was carried.
            state["this_order"] = self.this_order
        if hasattr(self, "last_sample"):
            state["last_sample"] = _out(self.last_sample)
        if hasattr(self, "timestep_list"):
            state["timestep_list"] = [
                None if t is None else (t.item() if hasattr(t, "item") else t)
                for t in self.timestep_list
            ]
        return state

    def restore_state(self, state) -> None:
        import torch as _torch

        device = self.sigmas.device if getattr(self, "sigmas", None) is not None else None

        def _back(a):
            if a is None:
                return None
            t = _torch.from_numpy(a)
            return t.to(device) if device is not None else t

        self._step_index = state.get("_step_index")
        self.lower_order_nums = state.get("lower_order_nums", 0)
        self.model_outputs = [_back(a) for a in state.get("model_outputs", [])]
        if "this_order" in state:
            self.this_order = state["this_order"]
        if "last_sample" in state:
            self.last_sample = _back(state["last_sample"])
        if "timestep_list" in state:
            self.timestep_list = list(state["timestep_list"])
