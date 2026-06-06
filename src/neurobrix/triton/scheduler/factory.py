"""Triton scheduler factory — zero-torch mirror of core/module/scheduler/factory.py.

Maps the SAME config keys / _class_name aliases to the triton (NBXTensor)
schedulers. The triton flows call this; the PyTorch flows call the core factory.
Two totally separate factories, one config schema.
"""
from .dpm_solver_pp import TritonDPMSolverPPScheduler
from .flow_euler import TritonFlowEulerScheduler

# alias -> triton class
_ALIASES = {
    "DPMSolverMultistepScheduler": TritonDPMSolverPPScheduler,
    "DPMSolverSinglestepScheduler": TritonDPMSolverPPScheduler,
    "dpmsolver++": TritonDPMSolverPPScheduler,
    "dpm++": TritonDPMSolverPPScheduler,
    "dpm++_2m": TritonDPMSolverPPScheduler,
    "dpm++_2m_karras": TritonDPMSolverPPScheduler,
    "DDPMScheduler": TritonDPMSolverPPScheduler,  # VibeVoice 'ddpm' alias → DPM++ (matches core)
}


class TritonSchedulerFactory:
    @staticmethod
    def create(config: dict):
        name = config.get("_class_name") or config.get("scheduler") or "dpmsolver++"
        # Flow-matching Euler -> separate triton FlowEuler (NOT DPM++): the core
        # factory maps FlowMatchEulerDiscreteScheduler to the FlowEuler algorithm,
        # so the triton mirror must too (different algorithm; routing it to DPM++
        # would change Sana/Flex output). Matches core/module/scheduler/factory.py.
        if name in ("FlowMatchEulerDiscreteScheduler", "FlowEulerScheduler",
                    "flow_euler"):
            return TritonFlowEulerScheduler(config)
        cls = _ALIASES.get(name)
        if cls is None:
            raise RuntimeError(
                f"ZERO FALLBACK: no triton scheduler for '{name}'. "
                f"Known: {sorted(_ALIASES)} + flow_euler.")
        return cls(config)
