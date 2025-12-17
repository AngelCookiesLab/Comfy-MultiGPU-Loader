"""
Comfy MultiGPU Loader
Copyright (C) 2025 Stefan Felton-Glenn  <stefanfg@protonmail.com>

This program comes with ABSOLUTELY NO WARRANTY; see LICENSE for details.
This is free software, and you are welcome to redistribute it
under the terms of the GPL-3.0; see LICENSE.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw


class GPUStatusDisplay:
    """Displays live GPU VRAM utilisation and current shard layout if available."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model": ("MODEL",),
                "status": ("STRING", {"multiline": True}),
                "layout": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("status_text", "status_image")
    FUNCTION = "get_status"
    CATEGORY = "MultiGPU/Diagnostics"
    OUTPUT_NODE = True

    def get_status(self, model=None, status=None, layout=None):
        if not torch.cuda.is_available():
            status = "❌ No CUDA GPUs detected"
            img = self.create_status_viz([])
            return (status, img)

        if status:
            status_prefix = status.strip()
        else:
            available_gpus = list(range(torch.cuda.device_count()))
            status_lines = [f"🎮 Detected {len(available_gpus)} GPU(s):"]
            status_lines.extend(self._gpu_lines(available_gpus))
            status_prefix = "\n".join(status_lines)

        layout_lines = []
        if layout:
            layout_lines.append(layout.strip())
        else:
            layout_lines.extend(self._model_layout_lines(model))

        combined_status = status_prefix
        if layout_lines:
            combined_status += "\n\nShard layout:\n" + "\n".join(layout_lines)

        img = self.create_status_viz(list(range(torch.cuda.device_count())))
        return (combined_status, img)

    def _gpu_lines(self, gpu_ids):
        lines = []
        for gpu_id in gpu_ids:
            props = torch.cuda.get_device_properties(gpu_id)
            used_gb, total_gb, pct = self._vram_usage(gpu_id)
            lines.append(f"GPU {gpu_id}: {props.name}")
            lines.append(f"  VRAM: {used_gb:.2f}GB / {total_gb:.2f}GB ({pct:.1f}%)")
        total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in gpu_ids) / 1024**3
        lines.append(f"💾 Total VRAM: {total_vram:.2f}GB")
        return lines

    def _model_layout_lines(self, model):
        if model is None or not hasattr(model, "_multi_gpu_device_map"):
            return []
        device_map = getattr(model, "_multi_gpu_device_map", {})
        if not device_map:
            return []
        gpu_ids = getattr(model, "_multi_gpu_gpu_ids", None)
        lines = []
        if gpu_ids:
            lines.append(f"GPUs: {', '.join(str(g) for g in gpu_ids)}")
        for block, device in device_map.items():
            lines.append(f"{block} → {device}")
        return lines

    def _vram_usage(self, gpu_id):
        try:
            free, total = torch.cuda.mem_get_info(gpu_id)
            used = total - free
        except RuntimeError:
            used = torch.cuda.memory_reserved(gpu_id)
            total = torch.cuda.get_device_properties(gpu_id).total_memory
        pct = (used / total) * 100 if total else 0.0
        used_gb = used / 1024**3
        total_gb = total / 1024**3
        return used_gb, total_gb, pct

    def create_status_viz(self, gpu_ids):
        """Creates visual GPU status bars."""
        if not gpu_ids:
            img_array = np.zeros((256, 512, 3), dtype=np.uint8)
            img_array[:, :] = [64, 64, 64]
        else:
            img_array = np.zeros((256, 512, 3), dtype=np.uint8)
            img_array[:, :] = [40, 40, 80]

            img = Image.fromarray(img_array)
            draw = ImageDraw.Draw(img)

            bar_height = 40
            spacing = 10
            start_y = 20

            for i, gpu_id in enumerate(gpu_ids):
                y = start_y + i * (bar_height + spacing)

                used_gb, total_gb, usage_pct = self._vram_usage(gpu_id)

                draw.rectangle(
                    [(20, y), (492, y + bar_height)],
                    fill=(60, 60, 100),
                    outline=(100, 100, 150),
                )

                usage_width = int((492 - 20) * (usage_pct / 100))
                if usage_width > 0:
                    draw.rectangle(
                        [(20, y), (20 + usage_width, y + bar_height)],
                        fill=(100, 200, 100),
                    )

                text = f"GPU {gpu_id}: {usage_pct:.1f}% ({used_gb:.2f}/{total_gb:.2f}GB)"
                draw.text((30, y + 10), text, fill=(255, 255, 255))

            img_array = np.array(img)

        img_array = img_array.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return img_tensor
