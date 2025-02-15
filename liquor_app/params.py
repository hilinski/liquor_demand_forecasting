import os
from pathlib import Path

# ACA VAN LAS CLAVES/ACCESOS, DEFINICIONES DE FLUJO, ETC
#VARIABLE EN MAYUSCULA
##################  VARIABLES  ##################
GCP_PUBLIC_DATA = os.environ.get("GCP_PUBLIC_DATA")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
ROOT_DIR = Path(__file__).parent.parent
RAW_DATA_PATH = Path(ROOT_DIR).joinpath("data","raw")
PROCESSED_DATA_PATH = Path(ROOT_DIR).joinpath("data","processed")
