# Current nodes

Loaders
- Load Checkpoint (MultiGPU) / Debug — all-in-one loader for SD/SDXL-style checkpoints.
- Load Flux Checkpoint (MultiGPU) / Debug — all-in-one loader for Flux; early/less tested; use debug node for diagnostics.

Sampling
- KSampler (MultiGPU) / Debug — extended sampler using the ComfyUI sampler API.

VAE
- VAE Decode (MultiGPU) / Debug — VAE decode with multi-GPU awareness.

Diagnostics
- GPU Status Display (+ Debug) — status and VRAM visualization.

Only the loader variants are “all-in-one” for convenience; other nodes mirror ComfyUI defaults with multi-GPU plumbing and diagnostics.
