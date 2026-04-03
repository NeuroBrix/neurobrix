"""NeuroBrix Triton Runtime — independent inference path.

Zero PyTorch dependency. All tensor ops via NBXTensor + Triton kernels.
Weight loading via safetensors numpy API + cudaMemcpy.

Entry point: TritonExecutor

Flow handlers (ported from core/flow/):
    TritonAutoregressiveHandler — autoregressive LLM generation
    TritonForwardPassHandler — sequential component execution
    TritonAudioEngine — audio STT/TTS orchestration
    TritonEncoderDecoderEngine — encoder-decoder (Whisper)
    TritonRNNTEngine — RNNT transducer (Parakeet)
    TritonTTSLLMEngine — TTS via LLM (Chatterbox)
    TritonDualAREngine — DualAR (OpenAudio)
    TritonStaticGraphHandler — single forward pass
"""
