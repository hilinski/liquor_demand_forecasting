import numpy as np
import pandas as pd
import pickle

import datetime

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler

from liquor_app.ml_logic.encoders import transform_numeric_features


def preprocess_features(X: pd.DataFrame, is_train:bool) -> tuple:

    def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Scikit-learn pipeline that transforms a cleaned dataset of shape (_, 7)
        into a preprocessed one of fixed shape (_, 65).

        Stateless operation: "fit_transform()" equals "transform()".
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
        numerical_features = ['week_year','week_of_year','bottles_sold']
        #numerical_features = ['week_year','week_of_year']
        num_pipe = make_pipeline(
            RobustScaler()
        )
        # COMBINED PREPROCESSOR

        final_preprocessor = ColumnTransformer(
            [
                ("cat_preproc", cat_pipe, categorical_features),
                ("num_preproc", num_pipe,  numerical_features)

            ],
            n_jobs=-1,
            remainder='passthrough'
        )

        return final_preprocessor

    print("\nPreprocessing features...")

    preprocessor = create_sklearn_preprocessor()

    if is_train:
        X_processed = preprocessor.fit_transform(X)
        # Guardar el preprocesador en un archivo
        with open("preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)

    else:
        with open("preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        X_processed = preprocessor.transform(X)

    col_names = preprocessor.get_feature_names_out()
    print("✅ X_processed, with shape", X_processed.shape)
    print(f'col_names from preprocessing before joins: {col_names}')

    return X_processed,col_names

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
