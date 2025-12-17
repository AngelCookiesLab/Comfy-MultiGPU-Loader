"""
Comfy MultiGPU Loader
Copyright (C) 2025 Stefan Felton-Glenn  <stefanfg@protonmail.com>

This program comes with ABSOLUTELY NO WARRANTY; see LICENSE for details.
This is free software, and you are welcome to redistribute it
under the terms of the GPL-3.0; see LICENSE.
"""

import torch

from ...utils import (
    get_available_gpus,
    validate_gpu_count,
    create_colored_image,
    DummyCLIP,
    DummyVAE,
    enable_unet_multigpu,
)

try:
    import folder_paths  # type: ignore
except ImportError:  # pragma: no cover - running outside ComfyUI
    folder_paths = None

try:
    import comfy.sd as comfy_sd  # type: ignore
    from comfy import model_management  # type: ignore
    import comfy.utils as comfy_utils  # type: ignore
    try:
        from comfy import cli_args as comfy_cli_args  # type: ignore
        comfy_args = getattr(comfy_cli_args, "args", None)
    except Exception:  # pragma: no cover - cli args may not exist in older builds
        comfy_cli_args = None
        comfy_args = None
except ImportError:  # pragma: no cover - running outside ComfyUI
    comfy_sd = None
    model_management = None
    comfy_utils = None
    comfy_cli_args = None
    comfy_args = None


class LoadCheckpointMultiGPUDebug:
    """
    Loads model checkpoints with optional distribution across multiple GPUs.
    Includes test mode to verify multi-GPU communication.
    """
    CATEGORY = "MultiGPU/Debug"
    clip_default_device = "auto"

    @classmethod
    def INPUT_TYPES(cls):
        ckpt_choices = cls._checkpoint_choices()
        default_ckpt = ckpt_choices[0] if ckpt_choices else "test_mode"
        clip_choices = cls._text_encoder_choices()
        vae_choices = cls._vae_choices()
        clip_type_choices = ["auto", "sdxl", "sd3", "flux", "hunyuan_video", "hidream", "hunyuan_image"]
        return {
            "required": {
                "ckpt_name": (ckpt_choices, {"default": default_ckpt, "tooltip": "Checkpoint to load from ComfyUI models/checkpoints."}),
                "num_gpus": ([1, 2, 3, 4, "Auto"], {"default": 2, "tooltip": "How many GPUs to allocate (use Auto for all available)."}),
                "test_mode": ("BOOLEAN", {"default": False, "tooltip": "Run synthetic smoke test instead of loading real models."}),
            },
            "optional": {
                "gpu_ids": ("STRING", {"default": "0,1,2,3", "multiline": False, "tooltip": "Comma separated GPU indices in execution order."}),
                "clip_name1": (clip_choices, {"default": clip_choices[0] if clip_choices else "<auto>", "tooltip": "Primary text encoder file (leave <auto> to use checkpoint embedded CLIP)."}),
                "clip_name2": (clip_choices, {"default": clip_choices[0] if clip_choices else "<auto>", "tooltip": "Secondary/paired text encoder file."}),
                "clip_loader_type": (clip_type_choices, {"default": "auto", "tooltip": "Decoder profile for dual-CLIP loader."}),
                "vae_name": (vae_choices, {"default": vae_choices[0] if vae_choices else "<auto>", "tooltip": "External VAE to use when checkpoint does not bundle one."}),
                "log_vram_snapshot": ("BOOLEAN", {"default": False, "tooltip": "Record per-GPU VRAM usage before/after layout for troubleshooting."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "IMAGE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "test_image", "status")
    FUNCTION = "load_checkpoint"
    CATEGORY = "MultiGPU/Debug"

    def load_checkpoint(
        self,
        ckpt_name,
        num_gpus,
        test_mode,
        gpu_ids="0,1,2,3",
        clip_name1="<auto>",
        clip_name2="<auto>",
        clip_loader_type="auto",
        vae_name="<auto>",
        log_vram_snapshot=False,
    ):
        available_gpus = get_available_gpus()

        if not available_gpus:
            status = "❌ No CUDA GPUs available!"
            error_img = create_colored_image([128, 128, 128], "No GPUs")
            return (None, None, None, error_img, status)

        # Parse GPU IDs
        try:
            requested_gpu_ids = [int(x.strip()) for x in gpu_ids.split(',')]
            valid_gpu_ids = [gid for gid in requested_gpu_ids if gid in available_gpus]
        except Exception:
            valid_gpu_ids = available_gpus

        actual_num_gpus = validate_gpu_count(num_gpus, valid_gpu_ids)
        used_gpus = valid_gpu_ids[:actual_num_gpus]

        # TEST MODE - Uses dummy objects for workflow testing
        debug_enabled = self._debug_enabled() or test_mode or ckpt_name == "test_mode"

        if test_mode or ckpt_name == "test_mode":
            test_image, test_status = self.pink_square_test(used_gpus)

            # Create REAL OBJECTS (not dicts!) that have the methods ComfyUI expects
            working_clip = DummyCLIP()
            working_vae = DummyVAE()

            dummy_model = {
                'type': 'multi_gpu_model_test',
                'gpu_ids': used_gpus,
                'test_mode': True
            }

            status = f"🧪 TEST MODE\n{test_status}\n"
            status += "✅ Dummy CLIP/VAE active - workflow functional but outputs are test data"

            return (dummy_model, working_clip, working_vae, test_image, status)

        # PRODUCTION MODE - Load real models and optionally distribute across GPUs
        try:
            model, clip, vae, status_image, status_string = self.load_real_checkpoint(
                ckpt_name,
                used_gpus,
                clip_name1=clip_name1,
                clip_name2=clip_name2,
                clip_loader_type=clip_loader_type,
                vae_name=vae_name,
                debug_enabled=debug_enabled,
                log_vram_snapshot=log_vram_snapshot,
            )
            if not debug_enabled:
                status_image = None
            return (model, clip, vae, status_image, status_string)
        except Exception as exc:
            status = f"❌ Failed to load checkpoint '{ckpt_name}': {exc}"
            error_img = create_colored_image([180, 70, 70], f"Load failed\n{ckpt_name}") if debug_enabled else None
            return (None, None, None, error_img, status)

    def pink_square_test(self, gpu_ids):
        """
        Tests multi-GPU communication by creating tensors on each GPU.
        Returns pink image if successful, gray if failed.
        """
        try:
            print(f"\n{'='*60}")
            print(f"🧪 MULTI-GPU TEST: Using GPUs {gpu_ids}")
            print(f"{'='*60}")

            gpu_results = []

            for gpu_id in gpu_ids:
                try:
                    device = f"cuda:{gpu_id}"

                    # Create test tensor on this specific GPU
                    test_tensor = torch.ones(100, 100, device=device) * (gpu_id + 1)
                    mean_val = test_tensor.mean().item()

                    # Verify tensor is actually on the correct GPU
                    assert test_tensor.device.type == "cuda"
                    assert test_tensor.device.index == gpu_id

                    # Get memory usage
                    mem_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2

                    gpu_results.append({
                        'id': gpu_id,
                        'success': True,
                        'mean': mean_val,
                        'memory': mem_allocated
                    })

                    print(f"✅ GPU {gpu_id}: Tensor created, mean={mean_val:.1f}, VRAM={mem_allocated:.1f}MB")

                except Exception as e:
                    gpu_results.append({
                        'id': gpu_id,
                        'success': False,
                        'error': str(e)
                    })
                    print(f"❌ GPU {gpu_id}: Failed - {e}")

            # Check if all GPUs succeeded
            all_success = all(r['success'] for r in gpu_results)

            if all_success:
                print(f"\n🎉 SUCCESS! All {len(gpu_ids)} GPUs responding correctly!")
                print(f"{'='*60}\n")

                text = f"✓ MultiGPU Success!\nUsing GPUs: {gpu_ids}"
                image = create_colored_image([255, 105, 180], text)  # Hot pink

                status = f"✅ SUCCESS!\nAll {len(gpu_ids)} GPUs communicating:\n"
                for r in gpu_results:
                    status += f"  GPU {r['id']}: {r['memory']:.1f}MB used\n"

                return image, status
            else:
                print(f"\n⚠️ Some GPUs failed the test")
                print(f"{'='*60}\n")

                failed = [r['id'] for r in gpu_results if not r['success']]
                image = create_colored_image([128, 128, 128], f"GPUs {failed} failed")  # Gray
                status = f"⚠️ Test failed on GPUs: {failed}"

                return image, status

        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            print(f"{'='*60}\n")

            error_image = create_colored_image([128, 128, 128], str(e))
            return error_image, f"❌ Error: {e}"

    @classmethod
    def _checkpoint_choices(cls):
        choices = ["test_mode"]
        if folder_paths is not None:
            try:
                choices.extend(folder_paths.get_filename_list("checkpoints"))
            except Exception:
                pass
        return choices

    @classmethod
    def _text_encoder_choices(cls):
        choices = ["<auto>"]
        if folder_paths is not None:
            try:
                choices.extend(folder_paths.get_filename_list("text_encoders"))
            except Exception:
                pass
        return choices

    @classmethod
    def _vae_choices(cls):
        choices = ["<auto>"]
        if folder_paths is not None:
            try:
                choices.extend(folder_paths.get_filename_list("vae"))
            except Exception:
                pass
        return choices

    def load_real_checkpoint(
        self,
        ckpt_name,
        gpu_ids,
        clip_name1="<auto>",
        clip_name2="<auto>",
        clip_loader_type="auto",
        vae_name="<auto>",
        debug_enabled=False,
        log_vram_snapshot=False,
    ):
        if comfy_sd is None or folder_paths is None:
            raise RuntimeError("ComfyUI core modules not available. Run this node inside ComfyUI.")

        if not gpu_ids:
            raise RuntimeError("No valid GPU IDs were supplied.")

        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        embedding_dirs = folder_paths.get_folder_paths("embeddings") if folder_paths else []

        print(f"📦 Loading checkpoint '{ckpt_name}' from {ckpt_path}")
        model, clip, vae, _ = comfy_sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=embedding_dirs,
        )

        status_notes = []
        vram_snapshots = {}

        clip_override_requested = clip_name1 != "<auto>" and clip_name2 != "<auto>"
        vae_override_requested = vae_name != "<auto>"

        if clip is None or clip_override_requested:
            if not clip_override_requested:
                raise RuntimeError("Checkpoint does not include a CLIP/text encoder. Please specify clip_name1 and clip_name2.")
            clip_type = clip_loader_type if clip_loader_type != "auto" else "flux"
            clip = self._load_dual_clip(clip_name1, clip_name2, clip_type, device_preference=self.clip_default_device)
            status_notes.append(f"Loaded external CLIP pair: {clip_name1}, {clip_name2} ({clip_type})")

        if vae is None or vae_override_requested:
            if vae_name == "<auto>":
                raise RuntimeError("Checkpoint does not include a VAE. Please specify vae_name.")
            vae = self._load_external_vae(vae_name)
            status_notes.append(f"Loaded external VAE: {vae_name}")

        try:
            model.model.eval()
        except AttributeError:
            pass

        primary_gpu = gpu_ids[0]
        device_map = None
        multi_gpu_notice = None

        if len(gpu_ids) > 1:
            try:
                if log_vram_snapshot:
                    vram_snapshots["before"] = self._capture_vram(gpu_ids)
                diffusion_core = self._locate_diffusion_module(model)
                if diffusion_core is None:
                    raise RuntimeError("Unable to locate diffusion model for multi-GPU setup.")
                device_map = enable_unet_multigpu(diffusion_core, gpu_ids)
                try:
                    setattr(diffusion_core, "_mg_model_owner", model)
                except Exception:
                    pass
                setattr(model, "_multi_gpu_usage_counts", {})
                print(f"✅ Multi-GPU layout applied: {device_map}")
                if log_vram_snapshot:
                    vram_snapshots["after"] = self._capture_vram(gpu_ids)
            except Exception as exc:
                multi_gpu_notice = f"⚠️ Multi-GPU setup failed: {exc}"
                device_map = None
                print(multi_gpu_notice)

        # Ensure the primary GPU is warmed up for CLIP/VAE (fallback if multi-GPU dispatch fails)
        target_device = torch.device(f"cuda:{primary_gpu}")
        try:
            if clip is not None and hasattr(clip, "cond_stage_model"):
                clip.cond_stage_model.eval()
                clip.cond_stage_model.to(target_device)
        except Exception as clip_exc:
            print(f"⚠️ Unable to move CLIP to GPU {primary_gpu}: {clip_exc}")
            try:
                clip.cond_stage_model.to("cpu")
            except Exception:
                pass
            status_notes.append("CLIP running on CPU due to GPU allocation limits")

        try:
            if vae is not None and hasattr(vae, "first_stage_model"):
                vae.first_stage_model.eval()
                vae.first_stage_model.to(target_device)
        except Exception as vae_exc:
            print(f"⚠️ Unable to move VAE to GPU {primary_gpu}: {vae_exc}")
            try:
                vae.first_stage_model.to("cpu")
            except Exception:
                pass
            status_notes.append("VAE running on CPU due to GPU allocation limits")

        # Force-eject/inject cycle so patches operate on the new device arrangement
        if model_management is not None and device_map is None:
            try:
                model_management.load_models_gpu([model], force_full_load=False)
            except Exception as manage_exc:
                print(f"⚠️ model_management.load_models_gpu warning: {manage_exc}")

        gpu_label = ", ".join(str(g) for g in gpu_ids)
        title = f"Loaded: {ckpt_name}\nGPUs: {gpu_label}"

        if device_map:
            setattr(model, "_multi_gpu_device_map", device_map)
            setattr(model, "_multi_gpu_gpu_ids", gpu_ids)

            if debug_enabled:
                status = "✅ Multi-GPU execution active\n"
                for key, dev in device_map.items():
                    status += f"  {key} → {dev}\n"
                status += f"Primary sampler device: cuda:{primary_gpu}\n"
            else:
                status = f"✅ Multi-GPU active on GPUs {gpu_label}"
        else:
            setattr(model, "_multi_gpu_device_map", None)
            setattr(model, "_multi_gpu_gpu_ids", [primary_gpu])

            if debug_enabled:
                status = f"✅ Checkpoint ready on GPU {primary_gpu}\n"
                if multi_gpu_notice:
                    status += f"{multi_gpu_notice}\n"
                    status += f"Requested GPUs: {gpu_label}\n"
            else:
                status = f"✅ Checkpoint ready on GPU {primary_gpu}"
                if multi_gpu_notice:
                    status += f" ({multi_gpu_notice.replace('⚠️ ', '')})"

        if status_notes:
            if debug_enabled:
                status += "\n" + "\n".join(f"• {note}" for note in status_notes)
            else:
                status += "; " + "; ".join(status_notes)

        if log_vram_snapshot and vram_snapshots:
            status += "\nVRAM snapshot (GB):"
            for phase, snapshot in vram_snapshots.items():
                status += f"\n  [{phase}] " + ", ".join(
                    f"GPU {gid}: {used:.2f}/{total:.2f}"
                    for gid, used, total in snapshot
                )

        status += "\nUse standard Comfy samplers/VAEs downstream."

        status_image = create_colored_image([70, 130, 180], title) if debug_enabled else None
        return model, clip, vae, status_image, status

    def _debug_enabled(self):
        if comfy_args is None:
            return False
        return getattr(comfy_args, "dev", False) or getattr(comfy_args, "debug", False)

    def _load_dual_clip(self, name1, name2, clip_type, device_preference="auto"):
        if comfy_sd is None or folder_paths is None:
            raise RuntimeError("ComfyUI core modules not available. Cannot load CLIP.")

        clip_type_key = clip_type.upper()
        if clip_type_key == "AUTO":
            clip_type_key = "FLUX"

        if not hasattr(comfy_sd.CLIPType, clip_type_key):
            raise RuntimeError(f"Unknown clip loader type '{clip_type}'.")

        clip_enum = getattr(comfy_sd.CLIPType, clip_type_key)

        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", name2)

        embedding_dirs = folder_paths.get_folder_paths("embeddings")
        model_options = {}
        if device_preference == "cpu":
            cpu_device = torch.device("cpu")
            model_options["load_device"] = cpu_device
            model_options["offload_device"] = cpu_device

        return comfy_sd.load_clip(
            ckpt_paths=[clip_path1, clip_path2],
            embedding_directory=embedding_dirs,
            clip_type=clip_enum,
            model_options=model_options,
        )

    def _load_external_vae(self, vae_name):
        if comfy_utils is None or comfy_sd is None or folder_paths is None:
            raise RuntimeError("ComfyUI core modules not available. Cannot load VAE.")

        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        vae_sd = comfy_utils.load_torch_file(vae_path)
        vae = comfy_sd.VAE(sd=vae_sd)
        vae.throw_exception_if_invalid()
        return vae

    def _capture_vram(self, gpu_ids):
        snapshot = []
        for gid in gpu_ids:
            try:
                free, total = torch.cuda.mem_get_info(gid)
                used = total - free
            except RuntimeError:
                used = torch.cuda.memory_allocated(gid)
                total = torch.cuda.get_device_properties(gid).total_memory
            snapshot.append(
                (
                    gid,
                    used / 1024**3,
                    total / 1024**3,
                )
            )
        return snapshot

    def _locate_diffusion_module(self, model_obj):
        """
        Walk the object graph to find the underlying diffusion model.
        Handles ModelPatcher, SD UNet, and Flux transformer layouts.
        """
        search_queue = [model_obj]
        visited = set()

        def enqueue(attr_owner):
            for attr in ("model", "diffusion_model", "inner_model", "flux_model"):
                if hasattr(attr_owner, attr):
                    candidate = getattr(attr_owner, attr)
                    if candidate is not None:
                        search_queue.append(candidate)

        enqueue(model_obj)

        while search_queue:
            current = search_queue.pop(0)
            if current is None:
                continue
            if id(current) in visited:
                continue
            visited.add(id(current))

            if hasattr(current, "input_blocks") and hasattr(current, "output_blocks"):
                return current
            if hasattr(current, "double_blocks") and hasattr(current, "single_blocks"):
                return current

            enqueue(current)

        return None


class LoadFluxCheckpointMultiGPUDebug(LoadCheckpointMultiGPUDebug):
    """
    Convenience loader preconfigured for Flux checkpoints.
    """

    CATEGORY = "MultiGPU/Debug"
    clip_default_device = "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        # Provide Flux-friendly defaults
        inputs["optional"]["clip_loader_type"] = (["flux"], {"default": "flux"})
        if "clip_name1" in inputs["optional"] and len(inputs["optional"]["clip_name1"][0]) > 1:
            flux_defaults = cls._default_flux_files()
            if flux_defaults:
                inputs["optional"]["clip_name1"] = (inputs["optional"]["clip_name1"][0], {"default": flux_defaults[0]})
                inputs["optional"]["clip_name2"] = (inputs["optional"]["clip_name2"][0], {"default": flux_defaults[1]})
        if "vae_name" in inputs["optional"]:
            flux_vae = cls._default_flux_vae()
            if flux_vae:
                inputs["optional"]["vae_name"] = (inputs["optional"]["vae_name"][0], {"default": flux_vae})
        return inputs

    @classmethod
    def _default_flux_files(cls):
        if folder_paths is None:
            return None
        try:
            clips = folder_paths.get_filename_list("text_encoders")
        except Exception:
            return None
        l_name = next((c for c in clips if "clip_l" in c.lower()), None)
        t5_name = next((c for c in clips if "t5" in c.lower()), None)
        if l_name and t5_name:
            return l_name, t5_name
        return None

    @classmethod
    def _default_flux_vae(cls):
        if folder_paths is None:
            return None
        try:
            vaes = folder_paths.get_filename_list("vae")
        except Exception:
            return None
        return next((v for v in vaes if "ae" in v.lower()), None)


class LoadCheckpointMultiGPU(LoadCheckpointMultiGPUDebug):
    CATEGORY = "MultiGPU/Loaders"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_checkpoint"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].pop("test_mode", None)
        inputs["optional"].pop("debug_info", None)
        inputs["optional"].pop("log_vram_snapshot", None)
        return inputs

    def load_checkpoint(
        self,
        ckpt_name,
        num_gpus,
        gpu_ids="0,1,2,3",
        clip_name1="<auto>",
        clip_name2="<auto>",
        clip_loader_type="auto",
        vae_name="<auto>",
    ):
        model, clip, vae, _image, _status = super().load_checkpoint(
            ckpt_name,
            num_gpus,
            False,
            gpu_ids,
            clip_name1,
            clip_name2,
            clip_loader_type,
            vae_name,
        )
        return (model, clip, vae)

    def _debug_enabled(self):
        return False


class LoadFluxCheckpointMultiGPU(LoadFluxCheckpointMultiGPUDebug):
    CATEGORY = "MultiGPU/Loaders"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_checkpoint"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].pop("test_mode", None)
        inputs["optional"].pop("debug_info", None)
        inputs["optional"].pop("log_vram_snapshot", None)
        return inputs

    def load_checkpoint(
        self,
        ckpt_name,
        num_gpus,
        gpu_ids="0,1,2,3",
        clip_name1="<auto>",
        clip_name2="<auto>",
        clip_loader_type="auto",
        vae_name="<auto>",
    ):
        model, clip, vae, _image, _status = super().load_checkpoint(
            ckpt_name,
            num_gpus,
            False,
            gpu_ids,
            clip_name1,
            clip_name2,
            clip_loader_type,
            vae_name,
        )
        return (model, clip, vae)

    def _debug_enabled(self):
        return False
