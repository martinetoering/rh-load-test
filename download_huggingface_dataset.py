import os

import requests
import zipfile
import io

def download_huggingface_dataset(download_link: str, local_dir: str):
    response = requests.get(download_link)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall(local_dir)

    # Ensure the local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    z.extractall(local_dir)

if __name__ == "__main__":
    # Usage example
    download_link = "https://huggingface.co/datasets/iix/mini_coco_linux/resolve/main/mini_coco_linux.zip"  # The download link from Hugginface
    local_dir = "datasets/mini_coco"  # Local path to save the files

    download_huggingface_dataset(download_link, local_dir)