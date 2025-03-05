import numpy as np
import pandas as pd
import pdb
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevents line wrapping

import os
import sys
sys.path.append(os.path.abspath(".."))

from pathlib import Path
from dateutil.parser import parse

import matplotlib.pyplot as plt

from liquor_app.params import *
from liquor_app.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from liquor_app.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from liquor_app.ml_logic.preprocessor import preprocess_features, create_sequences_fixed
from liquor_app.ml_logic.registry import load_model, save_model#, save_results
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


past_steps = 104
future_steps = 12


def get_data(min_date='2013-01-01', max_date='2025-01-31'):
    query = f"""
        WITH clean_data AS (
            SELECT * EXCEPT (store_number, zip_code, category, vendor_number, county_number),
                CAST(store_number AS NUMERIC) AS store_number,
                CAST(zip_code AS NUMERIC) AS zip_code,
                CAST(category AS NUMERIC) AS category,
                CAST(vendor_number AS NUMERIC) AS vendor_number
            FROM `bigquery-public-data.iowa_liquor_sales.sales`
            WHERE date >= '{min_date}' AND date <= '{max_date}'
            ORDER BY date ASC
        ),
        distinct_category AS (
            SELECT
                CAST(category AS NUMERIC) AS category,
                ARRAY_TO_STRING(ARRAY_AGG(category_name ORDER BY date DESC LIMIT 1), "") AS category_name
            FROM `bigquery-public-data.iowa_liquor_sales.sales`
            GROUP BY 1
        ),
        clean_data2 AS (
            SELECT cd.* EXCEPT (category_name),
                dc.category_name
            FROM clean_data cd
            LEFT JOIN distinct_category dc ON cd.category = dc.category
        ),
        group_and_others AS (
            SELECT date,
                -- **Remove county & group by category**
                CASE
                    WHEN category_name LIKE '%RUM%' THEN 'RUM'
                    WHEN category_name LIKE '%VODKA%' THEN 'VODKA'
                    WHEN category_name LIKE '%WHISK%' OR category_name LIKE '%SCOTCH%' THEN 'WHISKY'
                    WHEN category_name LIKE '%TEQUILA%' OR category_name LIKE '%MEZCAL%' THEN 'TEQUILA_MEZCAL'
                    WHEN category_name LIKE '%LIQUEUR%' THEN 'LIQUEURS'
                    WHEN category_name LIKE '%GIN%' THEN 'GIN'
                    ELSE 'OTROS'
                END AS category_name,
                SUM(bottles_sold) AS bottles_sold
            FROM clean_data2
            GROUP BY 1, 2
        ),
        combinations AS (
            SELECT *
            FROM UNNEST(GENERATE_DATE_ARRAY('{min_date}', '{max_date}', INTERVAL 1 DAY)) AS date
            CROSS JOIN (SELECT DISTINCT category_name FROM group_and_others) a
        ),
        data_combinations AS (
            SELECT c.*,
                DATE_TRUNC(c.date, WEEK) AS date_week,
                COALESCE(s.bottles_sold, 0) AS bottles_sold
            FROM combinations c
            LEFT JOIN group_and_others s
            ON c.date = s.date AND c.category_name = s.category_name
        )
        SELECT date_week, category_name,
            EXTRACT(YEAR FROM date_week) AS week_year,
            EXTRACT(WEEK(MONDAY) FROM date_week) AS week_of_year,
            SUM(bottles_sold) AS bottles_sold
        FROM data_combinations
        GROUP BY 1, 2, 3, 4
        ORDER BY category_name ASC, date_week ASC;


    """
    print(RAW_DATA_PATH)

    data = get_data_with_cache(
        gcp_project = GCP_PUBLIC_DATA,
        query = query,
        cache_path=Path(RAW_DATA_PATH).joinpath("data.csv"),
        data_has_header=True
    )

    return data


def assign_integer_ids(data: pd.DataFrame):
    """Assign unique integer IDs to categories instead of one-hot encoding."""

    category_mapping = {category: idx for idx, category in enumerate(data["category_name"].unique())}

    # Apply mapping
    data["category_id"] = data["category_name"].map(category_mapping).astype(int)

    print(f"✅ Assigned integer IDs for categories")

    return data, category_mapping




def preprocess(data):
    print("Running full preprocessing...")

    # Step 1️: Assign integer IDs (ONLY CATEGORY NOW)
    data, category_mapping = assign_integer_ids(data)

    # Step 2️: Drop unnecessary columns (Remove county-related columns)
    drop_cols = ["category_name", "date_week"]
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])

    # Step 3️: Convert numeric columns to float32
    numeric_cols = ["week_year", "week_of_year"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype(np.float32)

    # Step 4️: Ensure categorical IDs are int32 (ONLY CATEGORY)
    data["category_id"] = data["category_id"].astype(np.int32)  # ✅ Remove "county_id"

    # Step 5️: Apply `preprocess_features()` to transform data and get preprocessor
    processed_array, feature_names, preprocessor = preprocess_features(data, is_train=True)  # ✅ Return `preprocessor`

    # Convert processed NumPy array to DataFrame
    processed_data = pd.DataFrame(processed_array, columns=feature_names)

    print("✅ Preprocessing complete. Data is ready for model input.")

    return processed_data, category_mapping, preprocessor  # ✅ Return `preprocessor`



def train(data_processed, category_mapping, preprocessor):
    """Train the model on category-based sequences, removing county dependency."""

    # Convert `category_id` to NumPy arrays
    category_data = data_processed["remainder__category_id"].to_numpy(dtype=np.int32).reshape(-1, 1)

    # ✅ Use the correct column name for `bottles_sold`
    feature_columns = [
        "num_preproc__week_year", "num_preproc__week_sin", "num_preproc__week_cos",
        "num_preproc__bottles_sold",  # ✅ Ensure this exists
        "num_preproc__bottles_sold_4w_avg", "num_preproc__bottles_sold_12w_avg",
        "num_preproc__bottles_sold_diff"
    ]

    print("✅ Available columns in processed data:", data_processed.columns.tolist())  # Debugging step

    # ✅ Convert to numpy
    numeric_data = data_processed[feature_columns].to_numpy(dtype=np.float32)

    # ✅ Get `bottles_sold` index for target values
    bottles_sold_index = feature_columns.index("num_preproc__bottles_sold")

    # ✅ Create sequences
    X_dict, y = create_sequences_fixed(
        numeric_data, category_data,
        past_steps=past_steps, future_steps=future_steps,
        bottles_sold_index=bottles_sold_index
    )

    # ✅ Train/Validation Split (80/20)
    split_index = int(len(y) * 0.8)
    X_train = {key: value[:split_index] for key, value in X_dict.items()}
    X_val = {key: value[split_index:] for key, value in X_dict.items()}
    y_train, y_val = y[:split_index], y[split_index:]

    # ✅ Initialize Model
    model = initialize_model(
        input_shape=X_train["numeric_features"].shape[1:],
        num_categories=len(category_mapping)
    )

    # ✅ Compile & Train
    model = compile_model(model)
    model, history = train_model(model, X_train, y_train, validation_data=(X_val, y_val))

    # ✅ Save Model
    save_model(model=model)

    return model, X_val, y_val, category_mapping # ✅ Return preprocessor


def predict_on_validation(model, X_val, y_val, preprocessor):
    """
    Predicts future sales using the trained model and returns unscaled actual (y_val) and predicted (y_pred) values.
    """

    print("🔍 Generating validation predictions...")

    # 🔹 Predict future sales
    y_pred = model.predict(X_val)

    # Load the preprocessor and extract the scaler
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    # ✅ Extract the StandardScaler from the pipeline
    scaler = preprocessor.named_transformers_["num_preproc"].steps[0][1]  # First step in pipeline should be StandardScaler

    if not isinstance(scaler, StandardScaler):
        raise ValueError("🚨 Expected a StandardScaler but got:", type(scaler))

    # Get feature names and find the index of `bottles_sold`
    feature_names = preprocessor.get_feature_names_out()
    if "num_preproc__bottles_sold" not in feature_names:
        raise ValueError("🚨 'num_preproc__bottles_sold' not found in preprocessor feature names.")

    bottles_sold_index = list(feature_names).index("num_preproc__bottles_sold")

    # 🚀 Extract the `bottles_sold` column from scaler parameters
    bottles_sold_mean = scaler.mean_[bottles_sold_index]
    bottles_sold_std = scaler.scale_[bottles_sold_index]

    # ✅ Apply inverse scaling directly on `y_pred` and `y_val`
    y_pred_unscaled = (y_pred * bottles_sold_std) + bottles_sold_mean
    y_val_unscaled = (y_val * bottles_sold_std) + bottles_sold_mean

    print("✅ Model prediction completed!")
    print("🔍 y_pred shape:", y_pred.shape)
    print("🔍 y_val shape:", y_val.shape)

    return y_pred_unscaled, y_val_unscaled  # ✅ Return both for comparison





def plot_last_sequence_predictions(y_val, y_pred, X_val, category_mapping, category):
    """
    Plot the last known actual sales (y_val) and the next 12-week forecast (y_pred)
    for a specific category (statewide).
    """

    # 🔹 Find category ID
    category_id = category_mapping.get(category)

    if category_id is None:
        print(f"❌ Category '{category}' not found in mappings.")
        return

    # 🔹 Find last sequence for this category
    mask = (X_val["category_id"].flatten() == category_id)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        print(f"❌ No validation data found for category '{category}'")
        return

    last_idx = indices[-1]  # Get last sequence index

    # 🔹 Get last actual values and corresponding predictions
    last_actuals = y_val[last_idx]  # Last 12 actual weeks
    last_predictions = y_pred[last_idx]  # Next 12-week forecast

    # 🔹 Create week index
    past_weeks = np.arange(-12, 0)  # Last 12 weeks
    future_weeks = np.arange(0, 12)  # Next 12 weeks

    # 🔹 Plot actuals (last 12 weeks)
    plt.figure(figsize=(12, 6))
    plt.plot(past_weeks, last_actuals, marker='o', linestyle="-", color="blue", label="Actual Sales")

    # 🔹 Plot predictions (next 12 weeks)
    plt.plot(future_weeks, last_predictions, marker='x', linestyle="dashed", color="red", label="Predicted Sales")

    # Formatting
    plt.axvline(0, color="gray", linestyle="--", label="Prediction Start")  # Mark prediction start
    plt.xlabel("Weeks (Relative to Last Known Week)")
    plt.ylabel("Bottles Sold")
    plt.title(f"Sales Forecast for {category} (Statewide)")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    data = get_data()
    data_processed, county_mapping, category_mapping = preprocess(data)

    # Train Model
    model, X_val, y_val, _, _ = train(data_processed, county_mapping, category_mapping)
