import numpy as np
import pandas as pd
import pickle

import datetime
from liquor_app.params import *

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler, MinMaxScaler

from liquor_app.ml_logic.encoders import transform_numeric_features


import os
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline

def preprocess_features(X: pd.DataFrame, is_train: bool, county: object, category: object) -> tuple:
    """
    Preprocess features with separate preprocessor per county and category.
    Saves preprocessor as 'county-category-preprocessor.pkl'.
    """

    def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Scikit-learn pipeline that transforms a cleaned dataset into a preprocessed one.
        """
        # CATEGORICAL PIPE
        categorical_features = ['county', 'category_name']
        cat_pipe = make_pipeline(
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )

        # NUMERIC PIPE
        numerical_features = ['week_year', 'week_of_year', 'bottles_sold']
        num_pipe = make_pipeline(MinMaxScaler(feature_range=(0.1, 1)))

        # COMBINED PREPROCESSOR
        final_preprocessor = ColumnTransformer(
            [
                ("cat_preproc", cat_pipe, categorical_features),
                ("num_preproc", num_pipe, numerical_features)
            ],
            n_jobs=-1,
            remainder='passthrough'
        )

        return final_preprocessor

    print("\nPreprocessing features...")

    preprocessor = create_sklearn_preprocessor()


    # Clean county and category name for safe file naming (remove extra spaces, special characters)
    county_clean = county.replace(" ", "_").replace("-", "_").replace(",", "_").replace("'", "_")
    category_clean = category.replace(" ", "_").replace("-", "_").replace(",", "_").replace("'", "_")

    # Create processor path
    preprocessor_path = Path(PROCESOR_LOCAL_PATH).joinpath(f"{county_clean}-{category_clean}-preprocessor.pkl")

    # Debugging: Print the path to be used
    print(f"Saving/loading preprocessor at: {preprocessor_path}")

    if is_train:
        print("Entro")
        X_processed = preprocessor.fit_transform(X)
        # Save the preprocessor for the specific county and category
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)  # Ensure the directory exists
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)
        print(f"Preprocessor saved at: {preprocessor_path}")

    else:
        # Load the preprocessor for the specific county and category
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)
            print(f"Preprocessor loaded from: {preprocessor_path}")
        else:
            raise FileNotFoundError(f"Preprocessor file not found for {county} - {category} at {preprocessor_path}")

        X_processed = preprocessor.transform(X)

    col_names = preprocessor.get_feature_names_out()
    print("✅ X_processed, with shape", X_processed.shape)
    print(f'col_names from preprocessing before joins: {col_names}')

    return X_processed, col_names


# crear secuencias de RNN
def create_sequences(df, past_steps=52, future_steps=12):
    X, y = [], []
    df_x = df.copy()  # Keep all columns, including 'num_preproc__bottles_sold'
    df_y = df[["num_preproc__bottles_sold"]].copy()  # Target variable

    for i in range(len(df) - past_steps - future_steps):
        X.append(df_x.iloc[i : i + past_steps].values)  # Past data (including target)
        y.append(df_y.iloc[i + past_steps : i + past_steps + future_steps].values)  # Future target
    return np.array(X), np.array(y)

def create_sequences_padre(data_preproc, columnas_target, past_steps=10, future_steps=1):
    assert len(data_preproc) == len(columnas_target)
    df = pd.concat([data_preproc,columnas_target], axis='columns')
    X, y = [], []
    for county in data_preproc.iloc[:,data_preproc.columns.str.contains('cat_preproc__county_')].columns:
        for cat_prod in data_preproc.iloc[:,data_preproc.columns.str.contains('cat_preproc__category_name_')].columns:
            df_filtrado = df.query(f"{county} == 1 and {cat_prod} == 1")
            X_sequence, y_sequence = create_sequences(df_filtrado,past_steps,future_steps)
            for x_item in X_sequence:
                X.append(x_item)
            for y_item in y_sequence:
                y.append([y_item])
    return np.array(X), np.array(y)


def create_sequences_inference(data_preproc, past_steps=52):
    """
    Create sequences from new unseen data for inference (prediction).
    Returns only X_pred (input features), without y.
    """
    X_pred = []
    # Ensure that we have at least 'past_steps' weeks of data
    if len(data_preproc) < past_steps:
        raise ValueError(f"Not enough data. Need at least {past_steps} weeks, got {len(data_preproc)}")

    for county in data_preproc.iloc[:, data_preproc.columns.str.contains('cat_preproc__county_')].columns:
        for cat_prod in data_preproc.iloc[:, data_preproc.columns.str.contains('cat_preproc__category_name_')].columns:
            df_filtrado = data_preproc.query(f"{county} == 1 and {cat_prod} == 1")

            # Extract the last 'past_steps' weeks
            if len(df_filtrado) >= past_steps:
                X_pred.append(df_filtrado.iloc[-past_steps:].values)  # Last x weeks

    return np.array(X_pred)  # Shape: (num_groups, past_steps, num_features)

def create_sequences_inference_2(data_preproc, past_steps=52):
    """
    Create sequences from new unseen data for inference (prediction).
    Assumes data_preproc contains only one country-category combination.
    Returns only X_pred (input features), without y.
    """
    X_pred = []

    # Ensure that we have at least 'past_steps' weeks of data
    if len(data_preproc) < past_steps:
        raise ValueError(f"Not enough data. Need at least {past_steps} weeks, got {len(data_preproc)}")

    # Extract the last 'past_steps' weeks
    X_pred.append(data_preproc.iloc[-past_steps:].values)

    return np.array(X_pred)  # Shape: (1, past_steps, num_features)
