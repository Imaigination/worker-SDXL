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

device: str = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device == 'cuda' else torch.float32
# Setup the models
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=dtype, variant="fp16", use_safetensors=True
)
#pipe.to(device)
if device != 'cuda':
    #pipe.enable_xformers_amp()
    pipe.enable_attention_slicing()
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=dtype, use_safetensors=True, variant="fp16"
)
#refiner.to(device)
if device != 'cuda':
    #pipe.enable_xformers_amp()
    refiner.enable_attention_slicing()

def _save_and_upload_images(images, job_id):
    os.makedirs(f"{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"{job_id}", f"{index}.png")
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
    width = validated_input['validated_input']['width']
    height = validated_input['validated_input']['height']
    # Generate latent image using pipe
    print(f"Generating latent image for prompt: {prompt}")
    print(f"Using {num_inference_steps} inference steps")
    print(f"Generating {num_images_per_prompt} images per prompt")
    print(f"Image size: {width}x{height}")
    print(f'Validated input: {validated_input}')
    pipe_data = pipe(prompt=prompt,width=width, height=height,num_images_per_prompt=num_images_per_prompt, num_inference_steps=num_inference_steps , output_type="latent")
    print(f"Generated pipe_data: {pipe_data}")
    # Refine the image using refiner
    output = refiner(prompt=prompt,image=pipe_data.images).images

    image_urls = _save_and_upload_images(output, job['id'])

    return {"image_url": image_urls[0]} if len(image_urls) == 1 else {"images": image_urls}

runpod.serverless.start({"handler": generate_image})
