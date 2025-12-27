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
from itertools import cycle
import logging
import types
from typing import Any, Dict, Optional, Tuple
from einops import rearrange

try:
    from accelerate import dispatch_model
    _HAS_ACCELERATE = True
except ImportError:  # pragma: no cover - accelerate is optional
    dispatch_model = None
    _HAS_ACCELERATE = False

try:
    from comfy.ldm.modules.diffusionmodules import openaimodel  # type: ignore
except ImportError:  # pragma: no cover - running outside ComfyUI
    openaimodel = None

try:
    from comfy.ldm.flux.model import timestep_embedding as flux_timestep_embedding  # type: ignore
except ImportError:  # pragma: no cover
    flux_timestep_embedding = None


def get_available_gpus():
    """Returns list of available GPU indices"""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def validate_gpu_count(requested_gpus, available_gpus):
    """Validates requested GPU count against available"""
    if requested_gpus == "Auto":
        return len(available_gpus)
    
    requested = int(requested_gpus)
    if requested > len(available_gpus):
        print(f"⚠️ Requested {requested} GPUs but only {len(available_gpus)} available. Using {len(available_gpus)}.")
        return len(available_gpus)
    return requested


def create_colored_image(color, text, size=(512, 512)):
    """Creates a colored image with text"""
    img_array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img_array[:, :] = color
    
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, fill=(255, 255, 255))
    
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array)[None,]
    
    return img_tensor


class DummyCLIP:
    """
    Minimal CLIP wrapper for test mode.
    Provides the interface ComfyUI expects without real model.
    """
    def __init__(self):
        self.cond_stage_model = self
        self.tokenizer = self
        
    def tokenize(self, text):
        """Return dummy tokens that look like real CLIP tokens"""
        return torch.zeros(1, 77, dtype=torch.long)
    
    def encode_from_tokens(self, tokens, return_pooled=False):
        """Return dummy CLIP embeddings"""
        cond = torch.zeros(1, 77, 768)
        if return_pooled:
            pooled = torch.zeros(1, 768)
            return cond, pooled
        return cond
    
    def encode_from_tokens_scheduled(self, tokens, **kwargs):
        """
        Return dummy conditioning in the format ComfyUI expects.
        This is what CLIPTextEncode actually calls.
        """
        # ComfyUI expects a list of tuples: [(cond, pooled_output)]
        cond = torch.zeros(1, 77, 768)
        pooled = torch.zeros(1, 768)
        return [[cond, {"pooled_output": pooled}]]
    
    def encode(self, text):
        """Encode text directly (some nodes use this)"""
        tokens = self.tokenize(text)
        return self.encode_from_tokens_scheduled(tokens)


class DummyVAE:
    """
    Minimal VAE wrapper for test mode.
    Provides the interface ComfyUI expects without real model.
    """
    def __init__(self):
        pass
        
    def decode(self, samples):
        """Return dummy decoded image"""
        # samples['samples'] is typically (B, 4, H/8, W/8)
        # Output should be (B, H, W, 3) in range [0, 1]
        batch_size = samples['samples'].shape[0]
        height = samples['samples'].shape[2] * 8  # Upscale by 8x
        width = samples['samples'].shape[3] * 8
        
        # Return a dummy image (gradient pattern so you can see it worked)
        dummy_image = torch.zeros(batch_size, height, width, 3)
        
        # Create a simple gradient pattern
        for i in range(height):
            for j in range(width):
                dummy_image[0, i, j, 0] = i / height  # Red gradient
                dummy_image[0, i, j, 1] = j / width   # Green gradient
                dummy_image[0, i, j, 2] = 0.5         # Blue constant
        
        return dummy_image
    
    def encode(self, image):
        """Return dummy latent"""
        # image is (B, H, W, 3)
        batch_size = image.shape[0]
        height = image.shape[1] // 8
        width = image.shape[2] // 8
        
        # Return dummy latent
        return {'samples': torch.randn(batch_size, 4, height, width)}


def _module_device(module: torch.nn.Module, fallback: torch.device) -> torch.device:
    try:
        param = next(module.parameters())
        return param.device
    except StopIteration:
        return fallback


def _move_to_device_recursive(value: Any, device: torch.device) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device_recursive(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device_recursive(v, device) for v in value)
    if isinstance(value, dict):
        return {k: _move_to_device_recursive(v, device) for k, v in value.items()}
    return value


def _cached_tensor(cache: Dict[torch.device, torch.Tensor], base: torch.Tensor, device: torch.device) -> torch.Tensor:
    cached = cache.get(device)
    if cached is None:
        cache[device] = base.to(device)
        cached = cache[device]
    return cached


def _cached_structure(cache: Dict[torch.device, Any], value: Any, device: torch.device) -> Any:
    if value is None:
        return None
    cached = cache.get(device)
    if cached is None:
        cache[device] = _move_to_device_recursive(value, device)
        cached = cache[device]
    return cached


def _apply_control_multi_gpu(control: Optional[Dict[str, list]], name: str, tensor: torch.Tensor) -> torch.Tensor:
    if control is None or name not in control or len(control[name]) == 0:
        return tensor

    ctrl = control[name].pop()
    if ctrl is None:
        return tensor

    try:
        ctrl = _move_to_device_recursive(ctrl, tensor.device)
        tensor = tensor + ctrl
    except Exception:
        logging.warning("Failed to apply control '%s' on device %s", name, tensor.device)
    return tensor


def _tag_module_with_device(module: torch.nn.Module, device: torch.device):
    for sub in module.modules():
        setattr(sub, "_mg_device", device)
        if hasattr(sub, "comfy_cast_weights"):
            try:
                sub.comfy_cast_weights = True
            except Exception:
                pass


def _inject_multigpu_forward(unet: torch.nn.Module, handler):
    if hasattr(unet, "_mg_original_forward"):
        return

    target_attr = "_forward" if hasattr(unet, "_forward") else "forward"
    original_forward = getattr(unet, target_attr)

    def _forward_multi_gpu(self, *args, **kwargs):
        devices = getattr(self, "_mg_devices", None)
        if not devices or len(devices) <= 1:
            return original_forward(*args, **kwargs)
        usage_template = {str(device): 0 for device in devices}
        setattr(self, "_mg_runtime_usage", usage_template)
        return handler(self, original_forward, *args, **kwargs)

    unet._mg_original_forward = original_forward  # type: ignore[attr-defined]
    unet._mg_forward_attr = target_attr  # type: ignore[attr-defined]
    setattr(unet, target_attr, types.MethodType(_forward_multi_gpu, unet))


def _unet_forward_multigpu(unet, x, timesteps=None, context=None, y=None, control=None, transformer_options=None, extra_kwargs=None):
    if openaimodel is None:
        raise RuntimeError("ComfyUI diffusion modules not available. Multi-GPU execution must run inside ComfyUI.")

    devices = getattr(unet, "_mg_devices", None)
    if not devices or len(devices) <= 1:
        raise RuntimeError("Multi-GPU forward invoked without device assignments.")

    if transformer_options is None:
        transformer_options = {}
    if extra_kwargs is None:
        extra_kwargs = {}

    primary_device = devices[0]
    original_dtype = x.dtype
    x = x.to(primary_device)

    options = dict(transformer_options)
    options["original_shape"] = list(x.shape)
    options["transformer_index"] = 0
    transformer_patches = options.get("patches", {})

    num_video_frames = extra_kwargs.get("num_video_frames", getattr(unet, "default_num_video_frames", None))
    image_only_indicator = extra_kwargs.get("image_only_indicator", None)
    time_context = extra_kwargs.get("time_context", None)

    if timesteps is None:
        raise RuntimeError("timesteps must be provided for UNet forward pass.")
    timesteps = timesteps.to(primary_device)
    t_emb = openaimodel.timestep_embedding(timesteps, unet.model_channels, repeat_only=False).to(x.dtype).to(primary_device)
    emb = unet.time_embed(t_emb)

    emb_cache: Dict[torch.device, torch.Tensor] = {primary_device: emb}
    context_cache: Dict[torch.device, Any] = {}
    time_context_cache: Dict[torch.device, Any] = {}
    image_indicator_cache: Dict[torch.device, Any] = {}

    if unet.num_classes is not None:
        if y is None:
            raise RuntimeError("Label tensor 'y' required for class-conditional model.")
        y = y.to(primary_device)
        emb = emb + unet.label_emb(y)
        emb_cache[primary_device] = emb

    h = x
    hs: list[Tuple[torch.Tensor, torch.device]] = []

    def _prepare_inputs(target_device: torch.device):
        emb_local = _cached_tensor(emb_cache, emb, target_device)
        context_local = _cached_structure(context_cache, context, target_device)
        time_context_local = _cached_structure(time_context_cache, time_context, target_device)
        image_indicator_local = _cached_structure(image_indicator_cache, image_only_indicator, target_device)
        return emb_local, context_local, time_context_local, image_indicator_local

    usage_counts = getattr(unet, "_mg_runtime_usage", None)

    for idx, module in enumerate(unet.input_blocks):
        options["block"] = ("input", idx)
        target_device = getattr(module, "_mg_device", primary_device)
        h = h.to(target_device)
        emb_local, context_local, time_context_local, image_indicator_local = _prepare_inputs(target_device)

        h = openaimodel.forward_timestep_embed(
            module,
            h,
            emb_local,
            context_local,
            options,
            output_shape=None,
            time_context=time_context_local,
            num_video_frames=num_video_frames,
            image_only_indicator=image_indicator_local,
        )
        h = _apply_control_multi_gpu(control, "input", h)

        if "input_block_patch" in transformer_patches:
            for patch in transformer_patches["input_block_patch"]:
                h = patch(h, options)

        hs.append((h, target_device))
        if usage_counts is not None:
            key = str(target_device)
            usage_counts[key] = usage_counts.get(key, 0) + 1

        if "input_block_patch_after_skip" in transformer_patches:
            for patch in transformer_patches["input_block_patch_after_skip"]:
                h = patch(h, options)

    options["block"] = ("middle", 0)
    if unet.middle_block is not None:
        target_device = getattr(unet.middle_block, "_mg_device", primary_device)
        h = h.to(target_device)
        emb_local, context_local, time_context_local, image_indicator_local = _prepare_inputs(target_device)

        h = openaimodel.forward_timestep_embed(
            unet.middle_block,
            h,
            emb_local,
            context_local,
            options,
            output_shape=None,
            time_context=time_context_local,
            num_video_frames=num_video_frames,
            image_only_indicator=image_indicator_local,
        )
    h = _apply_control_multi_gpu(control, "middle", h)
    if usage_counts is not None and unet.middle_block is not None:
        middle_device = getattr(unet.middle_block, "_mg_device", primary_device)
        key = str(middle_device)
        usage_counts[key] = usage_counts.get(key, 0) + 1

    for idx, module in enumerate(unet.output_blocks):
        options["block"] = ("output", idx)
        hsp, _ = hs.pop()
        hsp = _apply_control_multi_gpu(control, "output", hsp)

        target_device = getattr(module, "_mg_device", primary_device)
        h = h.to(target_device)
        hsp = hsp.to(target_device)
        emb_local, context_local, time_context_local, image_indicator_local = _prepare_inputs(target_device)

        if "output_block_patch" in transformer_patches:
            for patch in transformer_patches["output_block_patch"]:
                h, hsp = patch(h, hsp, options)

        h = torch.cat([h, hsp], dim=1)
        del hsp

        if len(hs) > 0:
            output_shape = hs[-1][0].shape
        else:
            output_shape = None

        h = openaimodel.forward_timestep_embed(
            module,
            h,
            emb_local,
            context_local,
            options,
            output_shape=output_shape,
            time_context=time_context_local,
            num_video_frames=num_video_frames,
            image_only_indicator=image_indicator_local,
        )
        if usage_counts is not None:
            key = str(target_device)
            usage_counts[key] = usage_counts.get(key, 0) + 1

    h = h.to(dtype=original_dtype)
    out_device = _module_device(unet.out, primary_device)
    h = h.to(out_device)
    if usage_counts is not None:
        unet._mg_runtime_usage = usage_counts  # type: ignore[attr-defined]
        owner = getattr(unet, "_mg_model_owner", None)
        if owner is not None:
            setattr(owner, "_multi_gpu_usage_counts", dict(usage_counts))
    return unet.out(h)


def _unet_forward_handler(unet, original_forward, x, timesteps=None, context=None, y=None, control=None, transformer_options=None, **kwargs):
    return _unet_forward_multigpu(
        unet,
        x,
        timesteps=timesteps,
        context=context,
        y=y,
        control=control,
        transformer_options=transformer_options,
        extra_kwargs=kwargs,
    )


def _flux_forward_multigpu(unet, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None, transformer_options=None, extra_kwargs=None, original_forward=None):
    if openaimodel is None:
        raise RuntimeError("ComfyUI diffusion modules not available. Multi-GPU execution must run inside ComfyUI.")

    devices = getattr(unet, "_mg_devices", None)
    if not devices or len(devices) <= 1:
        raise RuntimeError("Multi-GPU forward invoked without device assignments.")

    if transformer_options is None:
        transformer_options = {}
    if extra_kwargs is None:
        extra_kwargs = {}

    primary_device = devices[0]
    original_dtype = x.dtype

    x = x.to(primary_device)
    timestep = timestep.to(primary_device)
    context = context.to(primary_device)

    if y is None:
        y = torch.zeros((x.shape[0], unet.params.vec_in_dim), device=primary_device, dtype=x.dtype)
    else:
        y = y.to(primary_device)

    guidance_tensor = guidance.to(primary_device) if guidance is not None else None
    attn_mask = extra_kwargs.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(primary_device)

    process_img_fn = getattr(unet, "process_img", None)
    if process_img_fn is None and hasattr(unet, "image_model"):
        process_img_fn = getattr(unet.image_model, "process_img", None)

    if process_img_fn is None:
        if original_forward is None:
            raise RuntimeError("Flux model does not expose process_img and no fallback forward provided.")
        return original_forward(
            x.to(primary_device),
            timestep.to(primary_device),
            context.to(primary_device),
            y=y,
            guidance=guidance,
            ref_latents=ref_latents,
            control=control,
            transformer_options=transformer_options,
            **extra_kwargs,
        )

    processed_refs = []
    if ref_latents is not None:
        for ref in ref_latents:
            processed_refs.append(ref.to(primary_device))
    else:
        processed_refs = None

    img, img_ids = process_img_fn(x)
    img_tokens = img.shape[1]

    if processed_refs is not None:
        h = 0
        w = 0
        index = 0
        ref_latents_method = extra_kwargs.get("ref_latents_method", "offset")
        for ref in processed_refs:
            if ref_latents_method == "index":
                index += 1
                h_offset = 0
                w_offset = 0
            elif ref_latents_method == "uxo":
                index = 0
                h_offset = ((h + (unet.patch_size // 2)) // unet.patch_size) * unet.patch_size + h
                w_offset = ((w + (unet.patch_size // 2)) // unet.patch_size) * unet.patch_size + w
                h += ref.shape[-2]
                w += ref.shape[-1]
            else:
                index = 1
                h_offset = 0
                w_offset = 0
                if ref.shape[-2] + h > ref.shape[-1] + w:
                    w_offset = w
                else:
                    h_offset = h
                h = max(h, ref.shape[-2] + h_offset)
                w = max(w, ref.shape[-1] + w_offset)

            kontext, kontext_ids = process_img_fn(ref, index=index, h_offset=h_offset, w_offset=w_offset)
            img = torch.cat([img, kontext], dim=1)
            img_ids = torch.cat([img_ids, kontext_ids], dim=1)

    txt_ids = torch.zeros((x.shape[0], context.shape[1], 3), device=primary_device, dtype=x.dtype)
    out = _flux_forward_internal_multigpu(
        unet,
        img,
        img_ids,
        context,
        txt_ids,
        timestep,
        y,
        guidance_tensor,
        control,
        transformer_options,
        attn_mask,
    )
    out = out[:, :img_tokens]
    result = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=unet.patch_size, pw=unet.patch_size)
    return result[:, :, : x.shape[2], : x.shape[3]].to(original_dtype)


def _flux_forward_internal_multigpu(unet, img, img_ids, txt, txt_ids, timesteps, y, guidance, control, transformer_options, attn_mask):
    devices = getattr(unet, "_mg_devices", None)
    primary_device = devices[0]

    patches = transformer_options.get("patches", {})
    patches_replace = transformer_options.get("patches_replace", {})

    if flux_timestep_embedding is None:
        raise RuntimeError("Flux utilities unavailable. Multi-GPU execution requires Flux modules to be importable.")

    img = unet.img_in(img)
    vec = unet.time_in(flux_timestep_embedding(timesteps, 256).to(img.dtype))
    if unet.params.guidance_embed and guidance is not None:
        vec = vec + unet.guidance_in(flux_timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + unet.vector_in(y[:, : unet.params.vec_in_dim])
    txt = unet.txt_in(txt)

    if "post_input" in patches:
        for p in patches["post_input"]:
            out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids})
            img = out["img"]
            txt = out["txt"]
            img_ids = out["img_ids"]
            txt_ids = out["txt_ids"]

    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = unet.pe_embedder(ids)
    else:
        pe = None

    emb_cache: Dict[torch.device, torch.Tensor] = {primary_device: vec}
    txt_cache: Dict[torch.device, torch.Tensor] = {primary_device: txt}
    img_cache: Dict[torch.device, torch.Tensor] = {primary_device: img}
    pe_cache: Dict[torch.device, Any] = {primary_device: pe} if pe is not None else {}
    attn_cache: Dict[torch.device, Any] = {primary_device: attn_mask} if attn_mask is not None else {}

    def _get_vec(target_device):
        return _cached_tensor(emb_cache, vec, target_device)

    def _get_pe(target_device):
        if pe is None:
            return None
        return _cached_tensor(pe_cache, pe, target_device)

    def _get_attn(target_device):
        if attn_mask is None:
            return None
        return _cached_tensor(attn_cache, attn_mask, target_device)

    blocks_replace = patches_replace.get("dit", {})

    current_img = img_cache[primary_device]
    current_txt = txt_cache[primary_device]

    usage_counts = getattr(unet, "_mg_runtime_usage", None)

    for i, block in enumerate(unet.double_blocks):
        target_device = getattr(block, "_mg_device", primary_device)
        if current_img.device != target_device:
            current_img = current_img.to(target_device)
        if current_txt.device != target_device:
            current_txt = current_txt.to(target_device)
        img_cache[target_device] = current_img
        txt_cache[target_device] = current_txt
        emb_local = _get_vec(target_device)
        pe_local = _get_pe(target_device)
        attn_local = _get_attn(target_device)

        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("double_block", i)](
                {
                    "img": current_img,
                    "txt": current_txt,
                    "vec": emb_local,
                    "pe": pe_local,
                    "attn_mask": attn_local,
                },
                {"original_block": block_wrap},
            )
            current_txt = out["txt"]
            current_img = out["img"]
        else:
            current_img, current_txt = block(img=current_img, txt=current_txt, vec=emb_local, pe=pe_local, attn_mask=attn_local)

        if usage_counts is not None:
            key = str(target_device)
            usage_counts[key] = usage_counts.get(key, 0) + 1

        if control is not None:
            control_i = control.get("input")
            if control_i is not None and i < len(control_i):
                add = control_i[i]
                if add is not None:
                    add = _move_to_device_recursive(add, target_device)
                    current_img[:, : add.shape[1], ...] += add

        img_cache[target_device] = current_img
        txt_cache[target_device] = current_txt

    if current_img.dtype == torch.float16:
        current_img = torch.nan_to_num(current_img, nan=0.0, posinf=65504, neginf=-65504)

    current_img = torch.cat((current_txt, current_img), 1)

    for i, block in enumerate(unet.single_blocks):
        target_device = getattr(block, "_mg_device", primary_device)
        if current_img.device != target_device:
            current_img = current_img.to(target_device)
        emb_local = _get_vec(target_device)
        pe_local = _get_pe(target_device)
        attn_local = _get_attn(target_device)

        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("single_block", i)](
                {
                    "img": current_img,
                    "vec": emb_local,
                    "pe": pe_local,
                    "attn_mask": attn_local,
                },
                {"original_block": block_wrap},
            )
            current_img = out["img"]
        else:
            current_img = block(current_img, vec=emb_local, pe=pe_local, attn_mask=attn_local)

        if usage_counts is not None:
            key = str(target_device)
            usage_counts[key] = usage_counts.get(key, 0) + 1

        if control is not None:
            control_o = control.get("output")
            if control_o is not None and i < len(control_o):
                add = control_o[i]
                if add is not None:
                    add = _move_to_device_recursive(add, target_device)
                    current_img[:, txt.shape[1] : txt.shape[1] + add.shape[1], ...] += add

    current_img = current_img[:, txt.shape[1] :, ...]
    current_img = current_img.to(primary_device)
    vec_primary = _cached_tensor(emb_cache, vec, primary_device)
    if usage_counts is not None:
        unet._mg_runtime_usage = usage_counts  # type: ignore[attr-defined]
        owner = getattr(unet, "_mg_model_owner", None)
        if owner is not None:
            setattr(owner, "_multi_gpu_usage_counts", dict(usage_counts))
    return unet.final_layer(current_img, vec_primary)


def _flux_forward_handler(unet, original_forward, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None, transformer_options=None, **kwargs):
    return _flux_forward_multigpu(
        unet,
        x,
        timestep=timestep,
        context=context,
        y=y,
        guidance=guidance,
        ref_latents=ref_latents,
        control=control,
        transformer_options=transformer_options,
        extra_kwargs=kwargs,
        original_forward=original_forward,
    )


def distribute_unet_across_gpus(unet, gpu_ids):
    """
    Split the UNet across GPUs using accelerate.dispatch_model.

    Returns the device map that was applied, or None if single GPU.
    """
    if not isinstance(gpu_ids, (list, tuple)) or len(gpu_ids) <= 1:
        return None

    if not _HAS_ACCELERATE:
        raise ImportError("accelerate is required for multi-GPU dispatch but is not installed.")

    if unet is None:
        raise ValueError("UNet model is None, cannot distribute across GPUs.")

    device_strings = [f"cuda:{gid}" for gid in gpu_ids]
    device_iter = cycle(device_strings)

    device_map = {}

    # Time embedding and input conv stay on the first GPU to reduce transfers.
    device_map["time_embed"] = device_strings[0]
    if hasattr(unet, "input_blocks"):
        for idx, _ in enumerate(unet.input_blocks):
            device_map[f"input_blocks.{idx}"] = next(device_iter)
    if hasattr(unet, "middle_block"):
        device_map["middle_block"] = device_strings[-1]
    if hasattr(unet, "output_blocks"):
        for idx, _ in enumerate(unet.output_blocks):
            device_map[f"output_blocks.{idx}"] = next(device_iter)

    # Final output normalization / convolution should land back on GPU 0.
    for attr in ("out", "out_norm", "out_conv"):
        if hasattr(unet, attr):
            device_map[attr] = device_strings[0]

    # SDXL-style conditioning embeds; keep the entire block on the first GPU.
    if hasattr(unet, "label_emb"):
        device_map["label_emb"] = device_strings[0]
        for sub_name, _ in unet.label_emb.named_modules():
            if sub_name:
                device_map[f"label_emb.{sub_name}"] = device_strings[0]

    dispatch_model(unet, device_map=device_map)
    return device_map


def enable_unet_multigpu(unet: torch.nn.Module, gpu_ids):
    """
    Assigns UNet blocks to the provided GPU IDs and injects a custom forward pass
    that migrates activations between devices at runtime.

    Returns a mapping of module names to device strings.
    """
    if hasattr(unet, "input_blocks") and hasattr(unet, "output_blocks"):
        return _enable_unet_style_multigpu(unet, gpu_ids)
    if hasattr(unet, "double_blocks") and hasattr(unet, "single_blocks"):
        return _enable_flux_multigpu(unet, gpu_ids)
    raise RuntimeError("Unsupported diffusion model architecture for multi-GPU sharding.")


def _enable_unet_style_multigpu(unet: torch.nn.Module, gpu_ids):
    if openaimodel is None:
        raise RuntimeError("ComfyUI diffusion modules not available. Multi-GPU requires running inside ComfyUI.")

    if not isinstance(gpu_ids, (list, tuple)) or len(gpu_ids) <= 1:
        raise ValueError("enable_unet_multigpu requires at least two GPU IDs.")

    device_list = [torch.device(f"cuda:{gid}") for gid in gpu_ids]
    device_map: Dict[str, str] = {}

    # Assign input blocks in a round-robin manner.
    input_devices = []
    for idx, block in enumerate(unet.input_blocks):
        device = device_list[idx % len(device_list)]
        block.to(device)
        _tag_module_with_device(block, device)
        input_devices.append(device)
        device_map[f"input_blocks.{idx}"] = str(device)

    # Middle block uses the last device by default.
    if unet.middle_block is not None:
        middle_device = device_list[-1]
        unet.middle_block.to(middle_device)
        _tag_module_with_device(unet.middle_block, middle_device)
        device_map["middle_block"] = str(middle_device)
    else:
        middle_device = device_list[-1]

    # Output blocks mirror the input block device order in reverse to align with skips.
    reversed_inputs = list(reversed(input_devices))
    for idx, block in enumerate(unet.output_blocks):
        device = reversed_inputs[idx % len(reversed_inputs)]
        block.to(device)
        _tag_module_with_device(block, device)
        device_map[f"output_blocks.{idx}"] = str(device)

    # Shared components remain on the primary device.
    primary_device = device_list[0]
    unet.time_embed.to(primary_device)
    _tag_module_with_device(unet.time_embed, primary_device)

    if hasattr(unet, "label_emb") and unet.label_emb is not None:
        unet.label_emb.to(primary_device)
        _tag_module_with_device(unet.label_emb, primary_device)

    unet.out.to(primary_device)
    _tag_module_with_device(unet.out, primary_device)

    unet._mg_devices = tuple(device_list)  # type: ignore[attr-defined]
    unet._mg_device_map = device_map  # type: ignore[attr-defined]

    _inject_multigpu_forward(unet, _unet_forward_handler)
    return device_map


def _enable_flux_multigpu(unet: torch.nn.Module, gpu_ids):
    if not isinstance(gpu_ids, (list, tuple)) or len(gpu_ids) <= 1:
        raise ValueError("enable_unet_multigpu requires at least two GPU IDs.")

    device_list = [torch.device(f"cuda:{gid}") for gid in gpu_ids]
    primary_device = device_list[0]
    device_map: Dict[str, str] = {}

    for attr in ["img_in", "time_in", "vector_in", "txt_in", "final_layer", "pe_embedder"]:
        module = getattr(unet, attr, None)
        if module is not None and hasattr(module, "to"):
            module.to(primary_device)
            _tag_module_with_device(module, primary_device)
            device_map[attr] = str(primary_device)

    guidance_module = getattr(unet, "guidance_in", None)
    if guidance_module is not None and hasattr(guidance_module, "to"):
        guidance_module.to(primary_device)
        _tag_module_with_device(guidance_module, primary_device)
        device_map["guidance_in"] = str(primary_device)

    double_devices = []
    for idx, block in enumerate(unet.double_blocks):
        device = device_list[idx % len(device_list)]
        block.to(device)
        _tag_module_with_device(block, device)
        device_map[f"double_blocks.{idx}"] = str(device)
        double_devices.append(device)

    for idx, block in enumerate(unet.single_blocks):
        device = device_list[idx % len(device_list)]
        block.to(device)
        _tag_module_with_device(block, device)
        device_map[f"single_blocks.{idx}"] = str(device)

    unet._mg_devices = tuple(device_list)  # type: ignore[attr-defined]
    unet._mg_device_map = device_map  # type: ignore[attr-defined]

    _inject_multigpu_forward(unet, _flux_forward_handler)
    return device_map
