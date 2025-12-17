# User Guide

## Model assets used in testing
- SD/SDXL: use the checkpoint’s built-in CLIP/VAE, or point the loader to an external VAE if preferred.
- Flux: external text encoders `clip_l.safetensors` and `t5xxl_fp8_e4m3fn_scaled.safetensors`, plus VAE `ae.safetensors` placed in `models/text_encoders` and `models/vae`, then selected in the Flux loader.

## Quick usage tips
- Use the debug loaders with `log_vram_snapshot` when diagnosing sharding; switch to non-debug loaders for production graphs.
- Pair the debug sampler with a display node to capture shard layout, per-GPU block counts, and VRAM deltas in `comfyui.log`.

## How to use (and debug)
- Use the *Load Checkpoint (MultiGPU) Debug* or *Load Flux Checkpoint (MultiGPU) Debug* node to load your model; set `gpu_ids` (e.g., `0,1,2,3`) and optionally enable `log_vram_snapshot` for before/after VRAM numbers.
- Pair with *KSampler (MultiGPU) Debug* to see shard layout, per-GPU block counts, VRAM deltas, and peaks during sampling. Connect its `status` output to a display node to log details into `comfyui.log`.
- For production graphs without extra outputs, use the non-debug loader/sampler counterparts.

## Debug helpers
- *Load Checkpoint (MultiGPU) Debug*: `log_vram_snapshot` appends per-GPU VRAM readings (in GB) before and after sharding to the node status and `comfyui.log`.
- *KSampler (MultiGPU) Debug*: reports shard counts per device, flags GPUs without assigned blocks, logs live VRAM deltas and peak usage per GPU, and lists how many blocks each GPU executed during the last sampling pass.
