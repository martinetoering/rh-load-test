import os
import logging
import requests
import argparse
import pandas as pd

def query(query_url: str, model_name: str, filename: str, filepath: str, height: int, width: int):
    with open(filepath, "rb") as f:
        data = f.read()
    
    token = os.getenv("TOKEN")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(query_url, data=data, headers=headers).json()
    logging.info(f"Response from {model_name} endpoint: {response}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Inference endpoint url",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Data directory",
        default="datasets/mini_coco/data"
    )
    parser.add_argument(
        "--data_list",
        type=str,
        help="Data csv",
        default="datasets/mini_coco/img_data.psv"
    )
    args = parser.parse_args()

    endpoint = f"/v2/models/{args.model_name}/infer"
    query_url = f"{args.url}{endpoint}"

    # Ensure the data directory exists
    if not os.path.exists(args.data_dir):
        raise ValueError("data dir not found")
    data_dir = args.data_dir

    # Load as pandas
    df = pd.read_csv(args.data_list, sep='|')
    df["filepath"] = [os.path.join(data_dir,f) for f in df.iloc[:, 0]]
    print(df)

    files = [os.path.join(data_dir,f) for f in df.iloc[:, 0] if os.path.isfile(os.path.join(data_dir,f))]
    logging.info(f"{len(files)} files found")

    for filename, height, width, filepath in zip(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, -1]):
        output = query(query_url, args.model_name, filename, filepath, height, width)
        break