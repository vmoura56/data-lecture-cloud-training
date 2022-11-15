from taxifare.ml_logic.params import LOCAL_REGISTRY_PATH

import glob
import os
import time
import pickle

from colorama import Fore, Style

from tensorflow.keras import Model, models
from google.cloud import storage


def save_local_model(model: Model = None,
               params: dict = None,
               metrics: dict = None) -> None:
    """
    persist trained model, params and metrics
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    print(Fore.BLUE + "\nSave model to local disk..." + Style.RESET_ALL)

    # save params
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        print(f"- params path: {params_path}")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        print(f"- metrics path: {metrics_path}")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    # save model
    if model is not None:
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", timestamp)
        print(f"- model path: {model_path}")
        model.save(model_path)

    print("\n✅ data saved locally")

    return None

def save_cloud_model(model):

    print("saving model in the cloud 🧬")

    BUCKET_NAME = "lecture_1039"

    storage_filename = "models/lecture_example"
    local_filename = "saved_model.pb"

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_filename)
    blob.upload_from_filename(local_filename)


def load_model(save_copy_locally=False) -> Model:
    """
    load the latest saved model, return None if no model found
    """
    print(Fore.BLUE + "\nLoad model from local disk..." + Style.RESET_ALL)

    # get latest model version
    model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")

    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None

    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")

    if os.environ["MODEL_TARGET"] == "local":

        model = models.load_model(model_path)
        print("\n✅ model loaded from disk")

    elif os.environ["MODEL_TARGET"] == "cloud":

        model_cloud_path = "gs://batch1039/models/20221115-105430"
        cloud_model = models.load_model(model_path)
        print("\n✅ model loaded from cloud")

    else:
        raise ValueError(f"Invalid .env config for model: {os.environ['MODEL_TARGET']} 🤕")

    return cloud_model
