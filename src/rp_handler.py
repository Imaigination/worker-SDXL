'''
Contains the handler function that will be called by the serverless.
'''

import os
import torch
import requests
from io import BytesIO
import json
import time
from PIL import Image
import boto3
import uuid
import botocore
import gc
import threading
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

import runpod
import base64
from runpod.serverless.utils import rp_upload, rp_cleanup
from diffusers.models import AutoencoderKL
from runpod.serverless.utils.rp_validator import validate
from dotenv import load_dotenv
from diffusers.schedulers import DDIMScheduler,LMSDiscreteScheduler,PNDMScheduler
from rp_schemas import INPUT_SCHEMA

load_dotenv()
models_dir =os.getenv("CACHE_DIR", "./models")
pipe_compile = os.getenv("PIPE_COMPILE", False)
small_width = 256
medium_width = 383
large_width = 512
print(f'Pipe compile = {pipe_compile}')
print(f'Using models dir: {models_dir}')
device: str = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"
dtype = torch.float16 if device == 'cuda' else torch.float32
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", cache_dir = models_dir)
# Setup the models
# scheduler ([`SchedulerMixin`]):
#             A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
#             [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    cache_dir = models_dir,
    vae=vae,
    torch_dtype=dtype, 
    variant="fp16",
    use_safetensors=True
    
)
# scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = scheduler
pipe.to(device, dtype)
# if pipe_compile:
#     pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
if device != 'cuda':
    pipe.enable_attention_slicing()

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    cache_dir = models_dir,
    vae =vae,
    torch_dtype=dtype,
    use_safetensors=True,
    variant="fp16"
)
refiner.to(device, dtype)
if device != 'cuda':
    #pipe.enable_xformers_amp()
    refiner.enable_attention_slicing()

def _save_and_upload_images(images, job_id):
    os.makedirs(f"{job_id}", exist_ok=True)
    
    # response = []
    paths= []
    # for index, image in enumerate(images):
    #     image_path = os.path.join(f"{job_id}", f"{index}.png")
    #     image.save(image_path)
    #     paths.append(image_path)

        # base64_images.append(base64.b64encode(image.tobytes()))
    print(paths)

    start = time.time()
    #image_urls = _upload_to_cloudflare(paths)    
    #image_urls = rp_upload.files(job_id, paths)
    image_urls = upload_images_v2(images)
    end = time.time()
    print(end - start)
    # print(image_urls)
    # print(f'URL={image_urls}')
    # response.append(image_urls)
    rp_cleanup.clean([f"/{job_id}"])
    # return base64_images
    #return image_urls
    return image_urls

def upload_images_v2(images):
    result = {}
    threads = []
    bucket_name = os.getenv('BUCKET_NAME')
        
    for index, image in enumerate(images):
        key = str(uuid.uuid4())

        result.pop(key, {})
        aratio = image.size[1] / image.size[0]
        thread = threading.Thread(target=upload_object_to_space, args=(image, bucket_name, key, 'full', result))
        # result.pop(key, {})
        result[key] = {}
        large = image.resize((large_width, int(medium_width * aratio)), Image.Resampling.LANCZOS)
        medium = image.resize((medium_width, int(medium_width * aratio)), Image.Resampling.LANCZOS)
        small = image.resize((small_width, int(small_width * aratio)), Image.Resampling.LANCZOS)
        
        thread.start()
        threads.append(thread)

        thread_large = threading.Thread(target=upload_object_to_space, args=(large, bucket_name, key, 'large', result))
        thread_large.start()
        threads.append(thread_large)

        thread_medium = threading.Thread(target=upload_object_to_space, args=(medium, bucket_name, key, 'medium', result))
        thread_medium.start()
        threads.append(thread_medium)

        thread_small = threading.Thread(target=upload_object_to_space, args=(small, bucket_name, key, 'small', result))
        thread_small.start()
        threads.append(thread_small)


    for thread in threads:
        thread.join()
        
    return result

def upload_object_to_space(image, bucket_name, object_key, postfix, result):
    aws_access_key_id=os.getenv('BUCKET_ACCESS_KEY_ID')
    aws_secret_access_key=os.getenv('BUCKET_SECRET_ACCESS_KEY')
    region = os.getenv('BUCKET_REGION')
    session = boto3.session.Session()
    output = BytesIO()
    image.save(output, format='png')    
    output.seek(0)
    image_name = f'{object_key}_{postfix}.png'
    client = session.client('s3',
                config=botocore.config.Config(s3={'addressing_style': 'virtual'}),
                region_name=region,
                endpoint_url=f'https://{region}.digitaloceanspaces.com',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key)
    client.put_object(Bucket=bucket_name,
                  Key=image_name,
                  Body=output.getvalue(),
                  ACL='public-read',
                  ContentType="image/png"
                  
                )
    domain = os.getenv('IMAGE_DOMAIN')
    image_url = f'{domain}/{image_name}'
    result[object_key][postfix] = image_url
    #result.append(image_url)

def upload_image(path, index, result):
    client_id = os.getenv("CLOUDFLARE_CLIENT_ID")
    api_key = os.getenv("CLOUDFLARE_API_KEY")
    url = f"https://api.cloudflare.com/client/v4/accounts/{client_id}/images/v1"
    payload = {}
    headers = {
        'Authorization': f"Bearer {api_key}"
    }
    files = [
        ('file', (f"{index}.png", open(path, 'rb'), 'image/png'))
    ]

    try:
        response = requests.post(url, headers=headers, data=payload, files=files)
        data = response.json()

        public_url = ""
        for url in data["result"]["variants"]:
            if "/public" in url:
                public_url = url
                break

        result.append(public_url)
    except Exception as e:
        print(f"Error uploading image: {e}")

def _upload_to_cloudflare(paths):
    result = []
    threads = []

    for index, path in enumerate(paths):
        thread = threading.Thread(target=upload_image, args=(path, index, result))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
        
    return result

def base64images(images, job_id):
    base64_images = []
    for img in images: 
        base64_images.append(base64.b64encode(img.tobytes()))
    rp_cleanup.clean([f"/{job_id}"])
    return base64_images

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
    refiner_strength = validated_input['validated_input']['refiner_strength']
    output = refiner(prompt,generator=generator, image=init_image,num_images_per_prompt=num_images_per_prompt,
        num_inference_steps = num_inference_steps,negative_prompt= negative_prompt, strength=strength, guidance_scale=guidance_scale).images
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

def load_lora_from_s3(lora_key, folder_to_save='lora'):
    os.makedirs(folder_to_save, exist_ok=True)
    name = lora_key.split('/')[-1]
    file_name = f'{folder_to_save}/{name}'
    if (os.path.isfile(file_name)):
        print('Lora file already existis. Returns cached')
        return name
    aws_access_key_id=os.getenv('BUCKET_ACCESS_KEY_ID')
    aws_secret_access_key=os.getenv('BUCKET_SECRET_ACCESS_KEY')
    region = os.getenv('BUCKET_REGION')
    bucket_name = os.getenv('BUCKET_NAME_LORA')
    session = boto3.session.Session()
   
    client = session.client('s3',
                config=botocore.config.Config(s3={'addressing_style': 'virtual'}),
                region_name=region,
                endpoint_url=f'https://{region}.digitaloceanspaces.com',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key)

    client.download_file(bucket_name, lora_key, file_name)
    return name

def text2text(job_input, job_id):
    gc.collect()
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
    negative_prompt = validated_input['validated_input']['negative_prompt']
    refiner_strength = validated_input['validated_input']['refiner_strength']
    guidance_scale = validated_input['validated_input']['guidance_scale']
    height = validated_input['validated_input']['height']
    base64 = validated_input['validated_input']['base64']
    # Generate latent image using pipe
    print(f"Generating latent image for prompt: {prompt}")
    print(f"Using {num_inference_steps} inference steps")
    print(f"Generating {num_images_per_prompt} images per prompt")
    print(f"Image size: {width}x{height}")
    print(f'Validated input: {validated_input}')
    lora_key = validated_input['validated_input']['lora_key']
    if lora_key:
        lora_file = load_lora_from_s3(lora_key)
        pipe.load_lora_weights(
            "lora/",
            weight_name=lora_file
        )
    with_refiner = refiner_strength > 0
    steps = num_inference_steps if not  with_refiner else num_inference_steps * (1 - refiner_strength)
    output_type = 'pil' if not with_refiner else 'latent'
    pipe_data = pipe(prompt=prompt,
                    width=width,
                    guidance_scale=guidance_scale,
                    negative_prompt= negative_prompt,
                    generator=generator,
                    height=height,
                    num_images_per_prompt=num_images_per_prompt,
                    num_inference_steps=steps,
                    output_type=output_type)
    print(f'Use refiner = {with_refiner}, output type = {output_type}')
    # Refine the image using refiner

    output = []
    if with_refiner:
        for img in pipe_data.images:
            output.append(refiner(
                prompt=prompt,
                generator=generator,
                negative_prompt= negative_prompt,
                guidance_scale=guidance_scale,
                strength=refiner_strength,
                num_inference_steps=num_inference_steps,
                image=img[None, :]
            ).images[0])
    else:
        output = pipe_data.images
    start_upload = time.time()
    if lora_key:
        pipe.unload_lora_weights()
    images = _save_and_upload_images(output,job_id) if not base64 else base64images(output, job_id)
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
        "upload_time": upload_time,
        "base64": base64
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