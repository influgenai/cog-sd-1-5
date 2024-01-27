from PIL import Image
from cog import BasePredictor, Input, Path
import torch
from typing import List
# from diffusers import DiffusionPipeline
# from diffusers import StableDiffusionPipeline
from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from torch.utils.data import Dataset
import os
import random
import io
import base64


# MODEL_ID = "runwayml/stable-diffusion-v1-5"
# MODEL_ID = "epicrealism.safetensors"
# MODEL_ID = "diffusers-cache/epicrealism.safetensors"
# MODEL_ID="https://civitai.com/api/download/models/143906?type=Model&format=SafeTensor&size=pruned&fp=fp16"
MODEL_CACHE = "diffusers-cache"

class Predictor(BasePredictor):
    def setup(self):
        print("Loading pipeline...")

        self.pipe = StableDiffusionPipeline.from_single_file(
            "diffusers-cache/epicrealism.safetensors",
            use_safetensors=True,
            load_safety_checker=False
        )

        lora_file_names = os.listdir(os.path.join(os.getcwd(), './diffusers-cache/loras'))
        lora_names = [item.split('.')[0] for item in lora_file_names]
        lora_weights = [6] * len(lora_names)

        for lora in lora_names:
            print(f"adding lora to model {lora}")
            self.pipe.load_lora_weights(f"./diffusers-cache/loras/{lora}.safetensors", adapter_name=lora)

        # self.pipe.set_adapters(lora_names, adapter_weights=lora_weights)
        # self.pipe.fuse_lora(adapter_names=lora_names)
        self.pipe.fuse_lora()

        self.pipe.to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="photo realistic, ElizabethTurner, <lora:frieren_v1:1>, aafrie, long hair, white hair, twintails, pointy ears, earrings, thick eyebrows, white capelet, striped shirt, long sleeves, belt, white skirt, black pantyhose, deep forest,",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="(CyberRealistic_Negative-neg:0.8), bicycle, nude, nsfw, large breasts, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, amputation",
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        # input_photo_path = input_photo
        # init_img = Image.open(input_photo_path)
        # init_img = init_img.resize((512, 512))
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)

        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            num_outputs=num_outputs,
            # scheduler=scheduler
        )

        output_paths = []
        
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))
        
        print("saved")
        
        return output_paths
    
def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]