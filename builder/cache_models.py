# builder/model_fetcher.py

import os
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

def get_diffusion_pipelines():
    print(F'TOKEN = {os.environ["HUGGING_FACE_HUB_TOKEN"]}')
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", 
                                                     torch_dtype=torch.float16, 
                                                     variant="fp16", 
                                                     use_auth_token=os.environ["HUGGING_FACE_HUB_TOKEN"],
                                                     use_safetensors=True)


    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9", 
                                                               torch_dtype=torch.float16, 
                                                               use_safetensors=True, 
                                                               use_auth_token=os.environ["HUGGING_FACE_HUB_TOKEN"],
                                                               variant="fp16")
    
    return pipe, refiner

if __name__ == "__main__":
    get_diffusion_pipelines()
