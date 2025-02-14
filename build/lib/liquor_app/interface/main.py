import numpy as np
import pandas as pd
import pdb

from pathlib import Path

from liquor_app.params import *
from liquor_app.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from liquor_app.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from liquor_app.ml_logic.preprocessor import preprocess_features
from liquor_app.ml_logic.registry import load_model, save_model, save_results

def preprocess(*args) -> None:

    query = f"""
                select *
                from `bigquery-public-data.iowa_liquor_sales.sales`
                where date <= '2023-03-31' and date >= '2023-01-01'
             """
    data = get_data_with_cache(gcp_project = GCP_PUBLIC_DATA,
        query = query,
        cache_path=Path(RAW_DATA_PATH).joinpath(f"data.csv"),
        data_has_header=True
    )
    return data


def train(*args) -> float:
    pass
def evaluate(*args) -> float:
    pass

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    pass

if __name__ == '__main__':
    preprocess()
    train()
    evaluate()
    pred()
