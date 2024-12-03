import os
from os import environ

import boto3

s3_endpoint_url = environ.get("S3_ENDPOINT") or environ.get("AWS_S3_ENDPOINT")
s3_access_key = environ.get("S3_ACCESS_KEY") or environ.get("AWS_ACCESS_KEY_ID")
s3_secret_key = environ.get("S3_SECRET_KEY") or environ.get("AWS_SECRET_ACCESS_KEY")
s3_bucket_name = environ.get("S3_BUCKET") or environ.get("AWS_S3_BUCKET")
s3_region = environ.get("AWS_DEFAULT_REGION", "None")


def move_s3_folder(s3_folder: str, new_s3_folder: str):
    # Initialize S3 client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        endpoint_url=s3_endpoint_url,
        region_name="eu-central-1",
    )

    # List objects within the specified S3 folder
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket_name, Prefix=s3_folder):
        if "Contents" in page:
            for obj in page["Contents"]:
                s3_key = obj["Key"]
                copy_source = {
                    'Bucket': s3_bucket_name,
                    'Key': s3_key,
                }
                base_filename = os.path.basename(s3_key)
                new_key = os.path.join(new_s3_folder, base_filename)
                print(f"Copy {s3_key} to {new_key}")
                s3_client.copy(copy_source, s3_bucket_name, new_key)


if __name__ == "__main__":
    # Usage example
    s3_folder = "huggingface_models_martine/onnx-community/gliner_multi-v2.1/onnx/"
    new_s3_folder = "huggingface_models_martine/onnx-community/gliner_multi-v2.1/onnx/1"

    move_s3_folder(s3_folder, new_s3_folder)