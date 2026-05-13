# Sana-4Kpx — verdict

**Verdict agent**: FAIL_EXPECTED
**Family**: image  •  **Mode**: t2i
**Duration**: 33.3s  •  **Exit**: 1
**Reason**: RuntimeError: Failed at op aten.convolution::55 (aten::convolution): CUDA ort of memory. Tried to allocate 36.00 GiB. GPU 2 has a total capacity of 31.74 GiB of which 13.73 GiB is free. Including non-PyTorch memory, this process has 18.00 GiB memory in use. Of the allocated memory 17.61 GiB is allocated by PyTorch, and 13.91 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

**Relaunch**:
```
/home/mlops/ml/venv/bin/neurobrix run --model Sana_1600M_4Kpx_BF16 --output /home/mlops/NeuroBrix_System/validation_outputs/runtime_alignment_phase5/Sana-4Kpx/output.png --prompt a red apple on a white plate --steps 2
```

Hocine validation: TODO
