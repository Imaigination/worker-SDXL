import requests
import tarfile
import replicate
import os

def download_safetensors_by_training_id(training_id):
    trainign_data = replicate.trainings.get(training_id)
    url = trainign_data.output['weights']
    r = requests.get(url, allow_redirects=True)
    path = training_id
    open(path, 'wb').write(r.content)

    #unzip tar archive
    folder_path = path+'_lora'
    tar = tarfile.open(path, "r:")
    os.makedirs(folder_path, exist_ok=True)
    tar.extractall(folder_path)
    tar.close()

    #remove tar archive
    os.remove(path)

    return f'{folder_path}'