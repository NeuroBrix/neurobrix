"""NeuroBrix Triton Runtime — independent inference path.

Zero PyTorch dependency. All tensor ops via NBXTensor + Triton kernels.
Weight loading via safetensors numpy API + cudaMemcpy.

Entry point: TritonExecutor
"""
