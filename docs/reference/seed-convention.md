# The seed convention — where a seed's noise comes from

**A NeuroBrix seed draws its initial noise from the stream of PyTorch's CPU generator**
(`torch.Generator("cpu").manual_seed(seed)`: MT19937 + Box-Muller), in both engines. The run's initial noise comes
from one seeded run stream in the nbxtensor library (`nbxtensor/random/rng_stream.py`, `nbxtensor/random/mt_randn.py`),
which reproduces that CPU stream without torch. The compiled branch and the Triton branch take the same initial
draw. A device generator (CUDA Philox, MPS) gives a different stream per platform and per vendor version.

The draws after the initial noise (the noise an ancestral or SDE scheduler adds at a step, a random operation in the
graph) come, on the Triton branch, from the same run stream; on the compiled branch, from a torch generator on the
run's device, seeded with the same seed.

A vendor example that draws its noise on the device (`torch.Generator("cuda")`, the usual model-card snippet)
renders another sample for the same integer seed **by construction, not by defect**. To compare the engine
with a vendor pipeline, use `torch.Generator("cpu").manual_seed(seed)` on the vendor's side, or hand the
vendor the engine's own initial noise: `NBX_DUMP_INIT_LATENT=<dir>` writes it.
