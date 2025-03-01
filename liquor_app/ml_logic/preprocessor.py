import numpy as np
import pandas as pd
import pickle

import datetime

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler

from liquor_app.ml_logic.encoders import transform_numeric_features



def assign_integer_ids(data: pd.DataFrame):
    """Assign unique integer IDs to categorical variables instead of one-hot encoding."""
    county_mapping = {county: idx for idx, county in enumerate(data["county"].unique())}
    category_mapping = {category: idx for idx, category in enumerate(data["category_name"].unique())}

    # Apply mappings
    data["county_id"] = data["county"].map(county_mapping).astype(int)
    data["category_id"] = data["category_name"].map(category_mapping).astype(int)

    print(f"✅ Assigned integer IDs for counties and categories")

    return data, county_mapping, category_mapping



def preprocess_features(X: pd.DataFrame, is_train: bool) -> tuple:
    """
    Preprocess features:
    - Convert county & category to integer IDs instead of one-hot encoding.
    - Scale numeric features.
    - Pass categorical IDs as is (for embeddings).
    """

    def create_sklearn_preprocessor() -> ColumnTransformer:
        # Keep only numeric features for scaling
        numerical_features = ["week_year", "week_of_year", "bottles_sold"]
        num_pipe = make_pipeline(RobustScaler())

        # No more one-hot encoding! Pass categorical IDs as they are.
        final_preprocessor = ColumnTransformer(
            [
                ("num_preproc", num_pipe, numerical_features)  # Only scale numeric features
            ],
            remainder="passthrough",  # Keep county_id & category_id unchanged
            n_jobs=-1
        )

        return final_preprocessor

    print("\nPreprocessing features...")

    preprocessor = create_sklearn_preprocessor()

    if is_train:
        X_processed = preprocessor.fit_transform(X)
        with open("preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)
    else:
        with open("preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        X_processed = preprocessor.transform(X)

    col_names = preprocessor.get_feature_names_out()
    print("✅ X_processed, with shape", X_processed.shape)
    print(f'col_names after preprocessing: {col_names}')

    return X_processed, col_names




# crear secuencias de RNN
def create_sequences(df, past_steps=10, future_steps=1):
    X, y = [], []
    df_x = df.drop(['bottles_sold'],axis='columns').copy()
    df_y = df[["bottles_sold"]].copy()
    for i in range(len(df) - past_steps - future_steps):
        X.append(df_x.iloc[i : i + past_steps].values)  # Past data
        y.append(df_y.iloc[i + past_steps : i + past_steps + future_steps]["bottles_sold"].values)  # Future target
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
                y.append(y_item)
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


def create_sequences_fixed(numeric_data, county_data, category_data, past_steps=10, future_steps=1):
    """
    Create sequences for RNN:
    - Uses numeric features (past sales & week info).
    - Keeps county_id & category_id separate for embeddings.
    - Ensures targets are correctly aligned.
    """

    X_numeric, X_county, X_category, y = [], [], [], []

    # 🚨 Iterate over dataset to extract sequences
    for i in range(len(numeric_data) - past_steps - future_steps):
        # Extract numeric features
        X_numeric.append(numeric_data[i : i + past_steps])

        # Extract categorical inputs (county & category) (🚨 Fix shape)
        X_county.append(county_data[i])  # Only take 1 value per sequence
        X_category.append(category_data[i])  # Only take 1 value per sequence

        # Extract targets (future sales values)
        y.append(numeric_data[i + past_steps : i + past_steps + future_steps, 0])  # First column = bottles_sold

    return {
        "numeric_features": np.array(X_numeric, dtype=np.float32),
        "county_id": np.array(X_county, dtype=np.int32).reshape(-1, 1),  # 🚨 Ensure shape (batch_size, 1)
        "category_id": np.array(X_category, dtype=np.int32).reshape(-1, 1)  # 🚨 Ensure shape (batch_size, 1)
    }, np.array(y, dtype=np.float32)
