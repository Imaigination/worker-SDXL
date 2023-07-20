'''
Contains the handler function that will be called by the serverless.
'''

import os
import torch
import time
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

import runpod
import base64
from runpod.serverless.utils import rp_upload, rp_cleanup
from diffusers.models import AutoencoderKL
from runpod.serverless.utils.rp_validator import validate
from dotenv import load_dotenv
from rp_schemas import INPUT_SCHEMA

load_dotenv()
device: str = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device == 'cuda' else torch.float32
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
# Setup the models
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9",vae=vae, torch_dtype=dtype, variant="fp16", use_safetensors=True
)
pipe.to(device)
if device != 'cuda':
    #pipe.enable_xformers_amp()
    pipe.enable_attention_slicing()
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-0.9",vae =vae, torch_dtype=dtype, use_safetensors=True, variant="fp16"
)
refiner.to(device)
if device != 'cuda':
    #pipe.enable_xformers_amp()
    refiner.enable_attention_slicing()

def _save_and_upload_images(images, job_id):
    os.makedirs(f"{job_id}", exist_ok=True)
    # base64_images = []
    response = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"{job_id}", f"{index}.png")
        image.save(image_path)
        image_url = rp_upload.upload_image(job_id, image_path)
        print(f'URL={image_url}')
        response.append(image_url)
        # base64_images.append(base64.b64encode(image.tobytes()))
    rp_cleanup.clean([f"/{job_id}"])
    return response

def img2img(job_input, job_id):
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    seed = validated_input['validated_input']['seed']
    if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
    generator = torch.Generator("cuda").manual_seed(seed)
    start = time.time()
    width = validated_input['validated_input']['width']
    height = validated_input['validated_input']['height']
    image_url = validated_input['validated_input']['init_image']
    init_image = load_image(image_url).convert("RGB")
    num_inference_steps = validated_input['validated_input']['num_inference_steps']
    num_images_per_prompt = validated_input['validated_input']['samples']
    negative_prompt= validated_input['validated_input']['negative_prompt']
    prompt = validated_input['validated_input']['prompt']
    strength = validated_input['validated_input']['strength']
    guidance_scale = validated_input['validated_input']['guidance_scale']
    output = refiner(prompt,generator=generator, image=init_image,num_images_per_prompt=num_images_per_prompt,num_inference_steps = num_inference_steps,negative_prompt= negative_prompt, strength=strength, guidance_scale=guidance_scale).images
    images = _save_and_upload_images(output,job_id)
    end = time.time()
    generation_time = end - start
    return {
        "images": images,
        "seed": seed,
        "prompt": prompt,
        "width": width,
        "height": height,
        "samples": num_images_per_prompt,
        "num_inference_steps": num_inference_steps,
        "generation_time": generation_time
    }

def text2text(job_input, job_id):
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    
    seed = validated_input['validated_input']['seed']
    if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
    start = time.time()
    generator = torch.Generator("cuda").manual_seed(seed)
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
    steps = num_inference_steps * 0.8
    pipe_data = pipe(prompt=prompt,width=width, generator=generator, height=height,num_images_per_prompt=num_images_per_prompt, num_inference_steps=steps , output_type="latent")
 
    # Refine the image using refiner

    output = []
    for img in pipe_data.images:
        output.append(refiner(prompt=prompt, generator=generator, num_inference_steps=num_inference_steps,image=img[None, :]).images[0])
    
    start_upload = time.time()
    images = _save_and_upload_images(output,job_id)
    end_upload = time.time()
    end = time.time()
    generation_time = end - start
    upload_time = end_upload - start_upload
    return {
        "images": images,
        "seed": seed,
        "prompt": prompt,
        "width": width,
        "height": height,
        "samples": num_images_per_prompt,
        "num_inference_steps": num_inference_steps,
        "generation_time": generation_time,
        "upload_time": upload_time
    }

def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]
    api_name = job_input['api_name']
    if api_name == 'text2text':
        return text2text(job_input, job['id'])
    elif api_name == 'img2img':
        return img2img(job_input, job['id'])
    # Input validation
    return text2text(job_input, job['id'])

runpod.serverless.start({"handler": generate_image})