"""Universal vision input preprocessing (mirror of core.module.audio).

`image_dsp` is the vendor-free numpy core shared by BOTH engines (R30/R34);
`input_processor.ImageInputProcessor` is the compiled-mode boundary that
converts the numpy result to a torch.Tensor. Design contract: the triton
path consumes the same numpy core through its own NBXTensor boundary; that
consumer lands with the first triton-side vision model (today the CLI
inputs dict carries the CPU tensor to both engines, as pre-migration).
"""
