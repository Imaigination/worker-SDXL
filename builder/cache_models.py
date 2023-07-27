# builder/model_fetcher.py

import os
import torch
import huggingface_hub
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.models import AutoencoderKL

from dotenv import load_dotenv

load_dotenv()
models_dir =os.getenv("CACHE_DIR", "/models")

def get_diffusion_pipelines():
    print(F'TOKEN = {os.environ["HUGGING_FACE_HUB_TOKEN"]}')
    from huggingface_hub.hf_api import HfFolder;
    HfFolder.save_token(os.environ["HUGGING_FACE_HUB_TOKEN"])
    vae = vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", cache_dir = models_dir)
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                                     torch_dtype=torch.float16, 
                                                     variant="fp16",
                                                     cache_dir = models_dir,
                                                     use_auth_token=os.environ["HUGGING_FACE_HUB_TOKEN"],
                                                     use_safetensors=True)


    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", 
                                                               torch_dtype=torch.float16, 
                                                               use_safetensors=True, 
                                                               cache_dir = models_dir,
                                                               use_auth_token=os.environ["HUGGING_FACE_HUB_TOKEN"],
                                                               variant="fp16")
    
    return pipe, refiner, vae

if __name__ == "__main__":
    get_diffusion_pipelines()
