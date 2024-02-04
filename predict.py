from PIL import Image
from cog import BasePredictor, Input, Path
import torch
from typing import List

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

MODEL_CACHE = "diffusers-cache"

class Predictor(BasePredictor):
    def setup(self):
        print("Loading pipeline...")

        self.pipe = StableDiffusionPipeline.from_single_file(
            "diffusers-cache/epicrealism.safetensors",
            use_safetensors=True,
            load_safety_checker=False
        )

        # lora_file_names = os.listdir(os.path.join(os.getcwd(), './diffusers-cache/loras'))
        # lora_names = [item.split('.')[0] for item in lora_file_names]
        # lora_weights = [0] * len(lora_names)

        # for lora in lora_names:
        #     print(f"adding lora to model {lora}")
        #     self.pipe.load_lora_weights(f"./diffusers-cache/loras/{lora}.safetensors", adapter_name=lora)

        # # This works   
        # self.pipe.set_adapters(lora_names, adapter_weights=lora_weights)
        
        # self.pipe.fuse_lora()

        self.pipe.to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="",
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[704, 768],
            default=704,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[896, 960],
            default=896,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=2,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=50, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=2.5
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
        )

        output_paths = []
        
        print(f"NUMBER OF OUTPUT IMAGES: {len(output.images)}")
        
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"

            # next 3 lines strip exif
            data = list(sample.getdata())
            image_without_exif = Image.new(sample.mode, sample.size)
            image_without_exif.putdata(data)
            image_without_exif.save(output_path)

            # as a good practice, close the file handler after saving the image.
            image_without_exif.close()

            #sample.save(output_path)
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