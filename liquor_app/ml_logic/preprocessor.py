import numpy as np
import pandas as pd
import pickle


from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def preprocess_features(X: pd.DataFrame, is_train: bool) -> tuple:
    """
    Preprocess features:
    - Convert categorical variables to integer IDs.
    - Apply sine/cosine encoding for `week_of_year`.
    - Scale numeric features, including `bottles_sold`.
    - Return processed data and preprocessor for inverse scaling.
    """

    # **Apply Sine & Cosine Transformation for `week_of_year`**
    X["week_sin"] = np.sin(2 * np.pi * X["week_of_year"] / 52)
    X["week_cos"] = np.cos(2 * np.pi * X["week_of_year"] / 52)

    # ✅ Include `bottles_sold` in scaling
    numerical_features = [
        "week_year", "week_sin", "week_cos",
        "bottles_sold",  # ✅ Now scaling this column
        "bottles_sold_4w_avg", "bottles_sold_12w_avg", "bottles_sold_diff"
    ]

    num_pipe = make_pipeline(StandardScaler())

    final_preprocessor = ColumnTransformer(
        [("num_preproc", num_pipe, numerical_features)],
        remainder="passthrough",  # Keep `category_id`
        n_jobs=-1
    )

    print("\nPreprocessing features...")

    if is_train:
        X_processed = final_preprocessor.fit_transform(X)
        with open("preprocessor.pkl", "wb") as f:
            pickle.dump(final_preprocessor, f)
    else:
        with open("preprocessor.pkl", "rb") as f:
            final_preprocessor = pickle.load(f)
        X_processed = final_preprocessor.transform(X)

    col_names = final_preprocessor.get_feature_names_out()
    print(f"✅ Processed Columns: {col_names}")  # ✅ Confirm `bottles_sold` is scaled

    return X_processed, col_names, final_preprocessor  # ✅ Return preprocessor






def create_sequences_fixed(numeric_data, category_data, past_steps=52, future_steps=12, bottles_sold_index=0):
    """
    Create sequences for RNN:
    - Uses numeric features (past sales & week info).
    - Keeps `category_id` separate for embeddings.
    - Ensures targets are correctly aligned.
    """

    X_numeric, X_category, y = [], [], []

    # 🚨 Iterate over dataset to extract sequences
    for i in range(len(numeric_data) - past_steps - future_steps):
        # Extract numeric features
        X_numeric.append(numeric_data[i : i + past_steps])

        # Extract categorical input (category_id)
        X_category.append(category_data[i])  # Only take 1 value per sequence

        # ✅ Extract targets (future sales values)
        y.append(numeric_data[i + past_steps : i + past_steps + future_steps, bottles_sold_index])

    return {
        "numeric_features": np.array(X_numeric, dtype=np.float32),
        "category_id": np.array(X_category, dtype=np.int32).reshape(-1, 1)  # 🚨 Ensure shape (batch_size, 1)
    }, np.array(y, dtype=np.float32)
