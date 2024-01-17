from PIL import Image
from cog import BasePredictor, Input, Path
import torch
from typing import List
from diffusers import DiffusionPipeline
from torch.utils.data import Dataset
import os
import random
import io
import base64


MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_CACHE = "diffusers-cache"

class Predictor(BasePredictor):
    def setup(self):
        print("Loading pipeline...")

        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            local_files_only=False,
        )

        for lora in os.listdir(os.path.join(os.getcwd(), './diffusers-cache/loras')):
            print(f"adding lora to model {lora}")
            self.pipe.load_lora_weights(f"./diffusers-cache/loras/{lora}")
            # self.pipe.fuse_lora(lora_scale = 0.5)

        self.pipe.to("cuda")
        
        self.pipe.enable_attention_slicing()

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="(Alessandra Ambrosio:0.5)",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
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
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        # scheduler: str = Input(
        #     default="DPMSolverMultistep",
        #     choices=[
        #         "DDIM",
        #         "K_EULER",
        #         "DPMSolverMultistep",
        #         "K_EULER_ANCESTRAL",
        #         "PNDM",
        #         "KLMS",
        #     ],
        #     description="Choose a scheduler.",
        # ),
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
        )

        # output = self.pipe(prompt=prompt, strength=0.75, guidance_scale=7.5, num_inference_steps=25 )
        output_paths = []
        
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))
        
        print("saved")
        
        return output_paths






    def predict(
        self,
        # input_photo: Path = Input(description="Path to the input photo"),
        prompt: str = Input(
            description="Input prompt",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
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
            description="Number of denoising steps", ge=1, le=500, default=50
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
        
        print("testing")
        # init_img = Image.open(input_photo_path)
        # init_img = init_img.resize((512, 512))

        output = self.pipe(prompt=prompt, strength=0.75, guidance_scale=7.5, num_inference_steps=25, )
        output_paths = []
        
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))
        
        print("saved")
        
        return output_paths