# Changelog

## 0.1.6 — 2025-12-18
- Marked project as sunset/as-is in README; roadmap frozen.
- No active support or new features; future work moves to a new ComfyUI-inspired project.

## 0.1.5 — 2025-12-18 
- Added `Hardware Validator (MultiGPU)` safety node to check GPU count/VRAM before heavy loads (e.g., Flux Dev 2 FP32).
- Loaders now accept a `safety_ok` input, allowing validation nodes to gate checkpoint loading.
- README/NODES updated with safety rail notes.
- Another not the Hardware Validator has not been tested a lot so stuff might be broken if this is true role back to 0.1.4

## 0.1.4 — 2025-12-17
- Documentation cleanup and modularization:
  - Added separate files for changelog, roadmap, node list, contributors, tested hardware, and user guide.
  - Trimmed README to link to the new docs; clarified node roles and usage links.
- Cleaned up GPL notice headers added/clarified across source files.

## 0.1.3 — 2025-12-17
- Switched license to GPL-3.0 for ComfyUI compatibility.
- Added GPL headers to source files.
- README cleanup and license references updated.

## 0.1.2 — 2025-12-17
- Continued license/header cleanup and documentation refinements.

## 0.1.1 — 2025-12-15
- Project folder cleanup.

## 0.1.0 — 2025-12-15
- Initial public-ready release:
  - Load Checkpoint (MultiGPU) / Debug
  - Load Flux Checkpoint (MultiGPU) / Debug
  - KSampler (MultiGPU) / Debug
  - VAE Decode (MultiGPU) / Debug
  - GPU Status Display (+ Debug)
  - Debug diagnostics (VRAM snapshots, shard layouts)
