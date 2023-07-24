# worker-SDXL
Serverless worker for runpod to run SDXL model

# Buidling
To build this serverless worker you need have access to SDXL 0.9 base and refiner repos.
Then you can building with this command:
```
docker build --build-arg HUGGING_FACE_HUB_TOKEN=YOURTOKENHERE
```

# run locally on runpod.io
1. Create env to store S3 info and huggingface token and open 8000 port during pod creation
2. 
2. apt-get update && apt-get install libgl1
4. python src/rp_handler.py --rp_serve_api --rp_api_host='0.0.0.0'
5. API Endpoind will be https://9mo9bjn982fg9r-8000.proxy.runpod.net/runsync where 9mo9bjn982fg9r is runpod id