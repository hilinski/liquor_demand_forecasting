import numpy as np
import pandas as pd

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

        categorical_features = ['county', 'category_name', 'vendor_name']

        cat_pipe = make_pipeline(
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )

        # COMBINED PREPROCESSOR
        #("num_preproc", num_pipe,  numerical_features)

        final_preprocessor = ColumnTransformer(
            [
                ("cat_preproc", cat_pipe, categorical_features)
            ],
            n_jobs=-1,
            remainder='passthrough'
        )

        return final_preprocessor

    print("\nPreprocessing features...")

    preprocessor = create_sklearn_preprocessor()

    if is_train:
        X_processed = preprocessor.fit_transform(X)


    else:
        X_processed = preprocessor.transform(X)

    col_names = preprocessor.get_feature_names_out()
    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed,col_names
