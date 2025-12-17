"""
Comfy MultiGPU Loader
Copyright (C) 2025 Stefan Felton-Glenn  <stefanfg@protonmail.com>

This program comes with ABSOLUTELY NO WARRANTY; see LICENSE for details.
This is free software, and you are welcome to redistribute it
under the terms of the GPL-3.0; see LICENSE.
"""

import torch

try:
    from nodes import common_ksampler  # type: ignore
except ImportError:  # pragma: no cover - when running outside ComfyUI
    common_ksampler = None

try:
    import comfy.samplers  # type: ignore

    _SAMPLER_CHOICES = tuple(comfy.samplers.KSampler.SAMPLERS)
    _SCHEDULER_CHOICES = tuple(comfy.samplers.KSampler.SCHEDULERS)
except ImportError:  # pragma: no cover
    _SAMPLER_CHOICES = ("euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde")
    _SCHEDULER_CHOICES = ("normal", "karras", "simple")


class KSamplerMultiGPU:
    """KSampler that defers to ComfyUI's native sampler while reporting shard layout."""

    @staticmethod
    def _memory_snapshot(gpu_ids):
        snapshot = {}
        if not isinstance(gpu_ids, (list, tuple)):
            return snapshot
        for gid in gpu_ids:
            try:
                free, total = torch.cuda.mem_get_info(gid)
                used = total - free
                snapshot[gid] = (used, total)
            except RuntimeError:
                snapshot[gid] = None
        return snapshot

    @staticmethod
    def _format_memory_deltas(before, after):
        lines = []
        activity = {}
        for gid, before_bytes in before.items():
            after_bytes = after.get(gid)
            if before_bytes is None or after_bytes is None:
                continue
            before_used, before_total = before_bytes
            after_used, after_total = after_bytes
            delta = after_used - before_used
            delta_gb = delta / 1024**3
            after_gb = after_used / 1024**3
            lines.append(
                f"cuda:{gid} Δ{delta_gb:+.2f}GB (now {after_gb:.2f}GB of {after_total / 1024**3:.2f}GB)"
            )
            activity[gid] = abs(delta_gb)
        return lines, activity

    @staticmethod
    def _reset_peak_stats(gpu_ids):
        if not isinstance(gpu_ids, (list, tuple)):
            return set()
        reset = set()
        for gid in gpu_ids:
            try:
                torch.cuda.reset_peak_memory_stats(gid)
                reset.add(gid)
            except RuntimeError:
                continue
        return reset

    @staticmethod
    def _collect_peak_stats(gpu_ids):
        peaks = {}
        if not isinstance(gpu_ids, (list, tuple)):
            return peaks
        for gid in gpu_ids:
            try:
                peak_bytes = torch.cuda.max_memory_allocated(gid)
                peaks[gid] = peak_bytes
            except RuntimeError:
                peaks[gid] = None
        return peaks

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (_SAMPLER_CHOICES,),
                "scheduler": (_SCHEDULER_CHOICES,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "status")
    FUNCTION = "sample"
    CATEGORY = "MultiGPU/Sampling"

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
    ):
        if common_ksampler is not None and not isinstance(model, dict):
            if negative is None:
                negative = positive
            gpu_ids = getattr(model, "_multi_gpu_gpu_ids", None)
            mem_before = self._memory_snapshot(gpu_ids)
            peak_tracked = self._reset_peak_stats(gpu_ids)
            latent_tuple = common_ksampler(
                model,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise,
            )
            latent_out = latent_tuple[0]

            device_map = getattr(model, "_multi_gpu_device_map", None)
            gpu_ids = getattr(model, "_multi_gpu_gpu_ids", None)
            mem_after = self._memory_snapshot(gpu_ids)
            peak_stats = self._collect_peak_stats(gpu_ids)
            usage_counts = getattr(model, "_multi_gpu_usage_counts", None)

            status_lines = [f"✅ Sampling complete → steps {steps}, CFG {cfg}"]
            if isinstance(gpu_ids, (list, tuple)):
                status_lines.append(f"GPUs: {', '.join(str(g) for g in gpu_ids)}")

            if device_map:
                status_lines.append("Shard layout:")
                for block, location in device_map.items():
                    status_lines.append(f"  {block} → {location}")
                if isinstance(gpu_ids, (list, tuple)):
                    shard_counts = {}
                    for device in device_map.values():
                        shard_counts[device] = shard_counts.get(device, 0) + 1
                    status_lines.append(
                        "Shard counts: "
                        + ", ".join(
                            f"{dev}: {count}"
                            for dev, count in sorted(shard_counts.items())
                        )
                    )
                    missing = [
                        gid
                        for gid in gpu_ids
                        if shard_counts.get(f"cuda:{gid}", 0) == 0
                    ]
                    if missing:
                        status_lines.append(
                            "⚠️ GPUs with no assigned blocks: "
                            + ", ".join(str(g) for g in missing)
                        )
            else:
                status_lines.append("Running on single GPU (no sharding active).")

            if mem_before and mem_after:
                delta_lines, delta_activity = self._format_memory_deltas(
                    mem_before, mem_after
                )
                if delta_lines:
                    status_lines.append("VRAM Δ during sampling:")
                    status_lines.extend(f"  {line}" for line in delta_lines)

            if usage_counts:
                status_lines.append("Runtime block executions:")
                zero_usage = []
                for device, count in sorted(usage_counts.items()):
                    status_lines.append(f"  {device}: {count}")
                    if count == 0:
                        zero_usage.append(device)
                if zero_usage:
                    status_lines.append(
                        "⚠️ GPUs with zero runtime executions: "
                        + ", ".join(zero_usage)
                    )
                try:
                    setattr(model, "_multi_gpu_usage_counts", {})
                except Exception:
                    pass

            if peak_stats:
                peak_lines = []
                idle_peaks = []
                for gid, peak_bytes in peak_stats.items():
                    if peak_bytes is None:
                        continue
                    peak_gb = peak_bytes / 1024**3
                    peak_lines.append(f"cuda:{gid} peak {peak_gb:.2f}GB")
                    if gid in peak_tracked and peak_gb < 0.05:
                        idle_peaks.append(gid)
                if peak_lines:
                    status_lines.append("Peak VRAM during sampling:")
                    status_lines.extend(f"  {line}" for line in peak_lines)
                if idle_peaks:
                    status_lines.append(
                        "⚠️ GPUs with low peak usage (<0.05GB): "
                        + ", ".join(str(g) for g in idle_peaks)
                    )

            status = "\n".join(status_lines)
            return (latent_out, status)

        warning = (
            "⚠️ Model not loaded with MultiGPU loader"
            if isinstance(model, dict)
            else "⚠️ Comfy sampler API unavailable in this environment"
        )
        dummy_latent = {"samples": torch.randn(1, 4, 64, 64)}
        return (dummy_latent, warning)
