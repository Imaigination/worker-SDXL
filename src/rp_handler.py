'''
Contains the handler function that will be called by the serverless.
'''

import os
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA

# Setup the models
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
refiner.to("cuda")

def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        image_url = rp_upload.upload_image(job_id, image_path)
        image_urls.append(image_url)
    rp_cleanup.clean([f"/{job_id}"])
    return image_urls

def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    
    prompt = validated_input['validated_input']['prompt']
    num_inference_steps = validated_input['validated_input']['num_inference_steps']
    num_images_per_prompt = validated_input['validated_input']['samples']
    width = validated_input['validated_input']['width'],
    height = validated_input['validated_input']['height'],
    # Generate latent image using pipe
    image = pipe(prompt=prompt,width=width, height=height,num_images_per_prompt=num_images_per_prompt, num_inference_steps=num_inference_steps , output_type="latent").images[0]

    # Refine the image using refiner
    output = refiner(prompt=prompt,width=width, height=height,num_images_per_prompt=num_images_per_prompt, num_inference_steps=num_inference_steps, image=image[None, :]).images[0]

    image_urls = _save_and_upload_images([output], job['id'])

    return {"image_url": image_urls[0]} if len(image_urls) == 1 else {"images": image_urls}

runpod.serverless.start({"handler": generate_image})
