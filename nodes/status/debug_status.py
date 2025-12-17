"""
Comfy MultiGPU Loader
Copyright (C) 2025 Stefan Felton-Glenn  <stefanfg@protonmail.com>

This program comes with ABSOLUTELY NO WARRANTY; see LICENSE for details.
This is free software, and you are welcome to redistribute it
under the terms of the GPL-3.0; see LICENSE.
"""

from .gpu_status import GPUStatusDisplay


class GPUStatusDisplayDebug(GPUStatusDisplay):
    CATEGORY = "MultiGPU/Debug"
