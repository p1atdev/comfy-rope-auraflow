from typing import Callable

import torch
import torch.nn as nn

from comfy.model_patcher import ModelPatcher
from comfy.ldm.aura.mmdit import MMDiT

from ..modules import replace_to_rope_modules


class AuraFlowRoPEPatcherNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "rope_theta": (
                    "INT",
                    {
                        "default": 10000,
                        "min": 0,
                        "max": 100000,
                        "step": 10000,
                        "display": "number",
                    },
                ),
                "dim_sizes": (
                    "STRING",
                    {
                        "default": "32, 112, 112",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "apply_patch"

    CATEGORY = "model_patches/unet"

    def parse_int_list(self, text: str) -> list[int]:
        return [int(x.strip()) for x in text.split(",")]

    def apply_patch(self, model: ModelPatcher, rope_theta: int, dim_sizes: str):
        dim_size_list = self.parse_int_list(dim_sizes)

        m = model.clone()
        diffusion_model: MMDiT = m.model.diffusion_model

        replace_to_rope_modules(diffusion_model)

        rope_args = {
            "rope_theta": rope_theta,
            "dim_sizes": dim_size_list,
        }
        for name, value in rope_args.items():
            m.model_options["transformer_options"][name] = value

        # m.set_model_unet_function_wrapper(rope_mmdit_forward)

        print(diffusion_model)

        return (m,)
