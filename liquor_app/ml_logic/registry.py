import glob
import os
import time
import pickle

from tensorflow import keras
from google.cloud import storage

from liquor_app.params import *


#def save_results(params: dict, metrics: dict) -> None:
#    """
#    Persist params & metrics locally on the hard drive at
#    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
#    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
#    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
#    """
#    timestamp = time.strftime("%Y%m%d-%H%M%S")
#
#    # Save params locally
#    if params is not None:
#        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
#        with open(params_path, "wb") as file:
#            pickle.dump(params, file)
#
#    # Save metrics locally
#    if metrics is not None:
#        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
#        with open(metrics_path, "wb") as file:
#            pickle.dump(metrics, file)
#
#    print("✅ Results saved locally")


def save_model(model: keras.Model = None, county='POLK', category='RUM') -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")



    # Save model locally
    model_path = os.path.join(MODEL_LOCAL_PATH, "models", f"{county}-{category}.h5")
    model.save(model_path)


    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    return None


def load_model(stage="Production", county = "Black Hawk", category = "GIN") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print(f"\nLoad latest model from local registry...")

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(MODEL_LOCAL_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*.h5")  # Ensure only .h5 models are selected

        if not local_model_paths:
            return None  # No models found

        # Convert county and category to uppercase (to match filename format)
        search_pattern = f"{county.upper()}-{category.upper()}"

        # Search for the matching model path
        latest_model_path = None
        for model_path in local_model_paths:
            filename = os.path.basename(model_path)  # Extract only the filename

            if search_pattern in filename:  # Check if county-category name exists in filename
                latest_model_path = model_path  # Store the latest matching model
                break  # Stop after finding the first match

        if latest_model_path:
            print(f"\nLoading model from: {latest_model_path}...")
            latest_model = keras.models.load_model(latest_model_path)
            print("✅ Model loaded from local disk")
            return latest_model

        print(f"❌ Model not found for: {search_pattern}")
        return None  # Ensure we return None only if no model was found


        #most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

       #print(f"\nLoad latest model from disk...")

        #latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        #print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print(f"\nLoad latest model from GCS...")

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(MODEL_LOCAL_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = keras.models.load_model(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None
