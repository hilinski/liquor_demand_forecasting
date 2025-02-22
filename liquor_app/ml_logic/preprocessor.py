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
def crear_secuencias(X, y, pasos=10):
    X, y = np.array(X), np.array(y)  # Convertir a arrays NumPy si aún no lo son
    secuencias_X = np.array([X[i:i+pasos] for i in range(len(X) - pasos)])
    secuencias_y = np.array([y[i+pasos] for i in range(len(y) - pasos)])
    return np.array(secuencias_X), np.array(secuencias_y)

# crear secuencias de RNN
def create_sequences(df, past_steps=10, future_steps=1):
    X, y = [], []
    for i in range(len(df) - past_steps - future_steps):
        X.append(df.iloc[i : i + past_steps].values)  # Past data
        y.append(df.iloc[i + past_steps : i + past_steps + future_steps]["num_preproc__bottles_sold"].values)  # Future target
    return np.array(X), np.array(y)
