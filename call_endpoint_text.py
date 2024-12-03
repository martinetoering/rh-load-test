import argparse
import json
import logging
import os

import requests
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning

from get_example_data import get_data

disable_warnings(category=InsecureRequestWarning)


def main(local: bool = False):
    data = {
        "text": (
            "Willem Alexander is de koning van Nederland. Hij houdt van"
            " sinaasappels en van Armin van Buren (een wereldberoemde DJ)"
        ),
        "labels": ["person", "country", "food", "dj"],
        "threshold": 0.5,
    }
    endpoint = "predict"

    if local:
        # lokaal volstaat het om met requests een post te doen omdat we hier geen token
        # op moeten halen om een call naar de API te kunnen doen.
        response = requests.post(
            verify=False, url=f"http://127.0.0.1:9000/{endpoint}", data=json.dumps(data)
        ).json()
    else:
        data = get_data()
        endpoint = os.getenv(
            "ENDPOINT",
            (
                "https://onnx-gliner-martine-demo-single-model.apps.rosa.rosa-jxx8z.wlcq.p3.openshiftapps.com/v2/models/gliner-multi/infer"
            ),
        )
        token = os.getenv("TOKEN")
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(endpoint, json=data, headers=headers).json()
    logging.info(f"Response from /{endpoint} endpoint: {response}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local",
        type=str,
        required=True,
        help="Whether to call local api or inference api",
    )
    args = parser.parse_args()
    local = True if args.local.lower() == "true" else False
    main(local=local)