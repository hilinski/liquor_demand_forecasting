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

from liquor_app.params import *
from liquor_app.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from liquor_app.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from liquor_app.ml_logic.preprocessor import preprocess_features, assign_integer_ids, create_sequences_inference, create_sequences_fixed
from liquor_app.ml_logic.registry import load_model, save_model#, save_results

past_steps = 52
future_steps = 12

def get_data(min_date='2013-01-01', max_date='2025-01-31'):
    query = f"""
        with clean_data as (
                select * EXCEPT (store_number, zip_code, category, vendor_number, county_number),
                CAST(store_number AS NUMERIC) as store_number ,
                CAST(zip_code AS NUMERIC) as zip_code ,
                CAST(category AS NUMERIC) as category,
                CAST(vendor_number AS NUMERIC) as vendor_number
                from `bigquery-public-data.iowa_liquor_sales.sales`
                where date >= '{min_date}' and date <= '{max_date}'
                --and CAST(vendor_number AS NUMERIC) in (260,421,65,370,85,434,35,301,259,115,395,55,420,205,380,192,297,300,255,389)
                ORDER BY date ASC
        ),
        distinct_vendor as (
                select
                CAST(vendor_number AS NUMERIC) as vendor_number,
                ARRAY_TO_STRING(ARRAY_AGG(vendor_name ORDER BY date DESC LIMIT 1),"") as vendor_name
                from `bigquery-public-data.iowa_liquor_sales.sales`
                group by 1
        ),
        distinct_category as (
                select
                CAST(category AS NUMERIC) as category,
                ARRAY_TO_STRING(ARRAY_AGG(category_name ORDER BY date DESC LIMIT 1),"") as category_name
                from `bigquery-public-data.iowa_liquor_sales.sales`
                group by 1
        ),
        distinct_store as (
                select
                CAST(store_number AS NUMERIC) as store_number,
                ARRAY_TO_STRING(ARRAY_AGG(store_name ORDER BY date DESC LIMIT 1),"") as store_name
                from `bigquery-public-data.iowa_liquor_sales.sales`
                group by 1
        ), clean_data2 as (
        select
                cd.* EXCEPT (vendor_name, category_name, store_name),
                dv.vendor_name,
                dc.category_name,
                ds.store_name
        from clean_data cd
        left join distinct_vendor dv on cd.vendor_number = dv.vendor_number
        left join distinct_category dc on cd.category = dc.category
        left join distinct_store ds on cd.store_number = ds.store_number
        ), group_and_others as (
        SELECT date,
        case when county in ('POLK','LINN','SCOTT','BLACK HAWK','JOHNSON') then county else 'OTHER' END AS county, #'POTTAWATTAMIE','DUBUQUE','STORY','WOODBURY','DALLAS'
        CASE
        WHEN category_name like '%RUM%' THEN 'RUM'
        WHEN category_name like '%VODKA%' THEN 'VODKA'
        WHEN category_name like '%WHISK%' or  category_name like '%SCOTCH%' THEN 'WHISKY'
        WHEN category_name like '%TEQUILA%' or category_name like '%MEZCAL%' THEN 'TEQUILA_MEZCAL'
        WHEN category_name like '%LIQUEUR%' THEN 'LIQUEURS'
        WHEN category_name like '%GIN%' THEN 'GIN'
        else 'OTROS'
        end as category_name,
        case when vendor_name in ('SAZERAC COMPANY  INC','DIAGEO AMERICAS','HEAVEN HILL BRANDS','LUXCO INC','JIM BEAM BRANDS','FIFTH GENERATION INC','PERNOD RICARD USA','MCCORMICK DISTILLING CO.','BACARDI USA INC','E & J GALLO WINERY') then vendor_name else 'OTHER' END as vendor_name,
        sum(bottles_sold) as bottles_sold
        FROM clean_data2
        group by 1,2,3,4
        ), summary as (
        select
        * EXCEPT (vendor_name)
        from group_and_others
        where lower(vendor_name) like '%bacardi%'
        ), combinations as (
        SELECT
          *
          FROM UNNEST(GENERATE_DATE_ARRAY('{min_date}', '{max_date}', INTERVAL 1 DAY)) as date
          cross join (select distinct category_name from summary) a
          cross join (select distinct county from summary) b
          ), data_combinations as (
        select c.*,
        date_trunc(c.date, WEEK) as date_week,
          coalesce(s.bottles_sold,0) as bottles_sold
          from combinations c
          left join summary s on c.date = s.date and c.category_name = s.category_name and c.county = s.county
          )
          select date_week, category_name, county,
          extract(YEAR FROM date_week) as week_year,
          extract(WEEK(MONDAY) from date_week) as week_of_year,
           sum(bottles_sold) as bottles_sold
           from data_combinations
           group by 1,2,3,4,5
           order by county asc, category_name asc, date_week asc

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
    """Assign unique integer IDs to categorical variables instead of one-hot encoding."""
    county_mapping = {county: idx for idx, county in enumerate(data["county"].unique())}
    category_mapping = {category: idx for idx, category in enumerate(data["category_name"].unique())}

    # Apply mappings
    data["county_id"] = data["county"].map(county_mapping).astype(int)
    data["category_id"] = data["category_name"].map(category_mapping).astype(int)

    print(f"✅ Assigned integer IDs for counties and categories")

    return data, county_mapping, category_mapping


def preprocess(data):
    print("🚀 Running preprocessing...")

    # Assign integer IDs first
    data, county_mapping, category_mapping = assign_integer_ids(data)

    # Drop unnecessary columns
    drop_cols = ["county", "category_name", "date_week"]  # Drop categorical text and date
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])

    # Convert numeric columns to float32 (fix dtype issues)
    numeric_cols = ["week_year", "week_of_year", "bottles_sold"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype(np.float32)

    # Ensure categorical IDs are int32
    data["county_id"] = data["county_id"].astype(np.int32)
    data["category_id"] = data["category_id"].astype(np.int32)

    return data, county_mapping, category_mapping



def train(data_processed, county_mapping, category_mapping):
    print("🔍 Processing training data...")

    # Convert `county_id` and `category_id` to NumPy arrays
    county_data = data_processed["county_id"].to_numpy(dtype=np.int32).reshape(-1, 1)
    category_data = data_processed["category_id"].to_numpy(dtype=np.int32).reshape(-1, 1)

    # Ensure bottles_sold is the first column before converting to numpy
    numeric_data = data_processed[["bottles_sold", "week_year", "week_of_year"]].to_numpy(dtype=np.float32)


    #Print shape checks before sequence creation
    print(f"🔍 numeric_data.shape: {numeric_data.shape}")
    print(f"🔍 county_data.shape: {county_data.shape}")
    print(f"🔍 category_data.shape: {category_data.shape}")

    # Create sequences
    # Create sequences
    X_dict, y = create_sequences_fixed(numeric_data, county_data, category_data, past_steps=52, future_steps=12)

    # Train/Validation Split (80/20)
    split_index = int(len(y) * 0.8)

    # Unpack dictionary properly
    X_train = {key: value[:split_index] for key, value in X_dict.items()}
    X_val = {key: value[split_index:] for key, value in X_dict.items()}
    y_train, y_val = y[:split_index], y[split_index:]

    print("🔍 y_train sample:\n", y_train[:10])
    print("🔍 y_val sample:\n", y_val[:10])

    # Print shapes before model training
    print(f"✅ X_train['numeric_features'].shape: {X_train['numeric_features'].shape}")  # (batch, 52, num_features)
    print(f"✅ X_train['county_id'].shape: {X_train['county_id'].shape}")  # (batch, 1)
    print(f"✅ X_train['category_id'].shape: {X_train['category_id'].shape}")  # (batch, 1)
    print(f"✅ y_train.shape: {y_train.shape}")

    # Initialize Model
    model = initialize_model(input_shape=X_train["numeric_features"].shape[1:],
                             num_counties=len(county_mapping),
                             num_categories=len(category_mapping))

    # Compile & Train
    model = compile_model(model)
    model, history = train_model(model, X_train, y_train, validation_data=(X_val, y_val))

    save_model(model=model)
    print("✅ Model training complete!")
    return model, X_val, y_val, county_mapping, category_mapping



def evaluate(*args) -> float:
    pass

def pred(X_pred:np.ndarray = None,  past_steps=past_steps, future_steps=future_steps) -> pd.DataFrame:

    if X_pred is None:
        query = """SELECT 'hello world'"""

        data = get_data_with_cache(GCP_PUBLIC_DATA,
            query,
            cache_path=Path(RAW_DATA_PATH).joinpath("data.csv"),
            data_has_header=True
        )
        df_pred = data.query(f"date_week >= '2024-01-01' and date_week <= '2024-12-31'")
        month_in_year = 12
        df_pred['date_week'] = pd.to_datetime(df_pred['date_week'])
        df_pred['num_month'] = df_pred['date_week'].dt.month
        df_pred['sin_MoSold'] = np.sin(2*np.pi*df_pred.num_month/month_in_year)
        df_pred['cos_MoSold'] = np.cos(2*np.pi*df_pred.num_month/month_in_year)
        df_pred = df_pred.drop('num_month', axis=1)
        print(" ✅ df_pred shape before process:", df_pred.shape)
        df_processed,col_names = preprocess_features(df_pred,False)
        print("✅ Data Proccesed ")

        df_processed = pd.DataFrame(
            df_processed,
            columns=col_names
        )
        print(" ✅ df_pred shape after process:", df_processed.shape)
        df_processed = df_processed.drop('remainder__date_week', axis=1)
        for columns in df_processed.columns:
            df_processed[columns] = df_processed[columns].astype(float)

        df_processed.columns = df_processed.columns.str.replace(" ", "_")
        print("df_processed info columns: ", df_processed.info())

        X_pred = create_sequences_inference(df_processed, past_steps=past_steps)
        print("✅ Create Sequences complete ")
        print(f"{X_pred[0][0]=}")

        print(f"cargando modelo")
        model = load_model()
        assert model is not None
        print(f"modelo cargado")
        y_pred = model.predict(X_pred)
        print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
        print("type of y_pred:" , type(y_pred))

        # 1. Get unique country-category pairs
        unique_pairs = df_pred[['county', 'category_name']].drop_duplicates().reset_index(drop=True)

        print("✅ Step 1 completed ")


        # 2. Create a DataFrame for predictions
        last_date = df_pred['date_week'].max()  # Get last available date
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=future_steps, freq='W')
        #print(f"{last_date=}")
        #print(f"{future_dates=}")
        #print(len(future_dates))

        print("✅ Step 2 completed ")

        # Create a DataFrame to store predictions
        predictions_df = pd.DataFrame()

        # 3. Fill the DataFrame with predictions
        for i, (county, category) in enumerate(unique_pairs.itertuples(index=False)):
            temp_df = pd.DataFrame({
                'date_week': future_dates,  # Assign future dates
                'county': county,
                'category_name': category,
                'bottles_sold': y_pred[i]  # Get the corresponding predictions
            })
            print(f"{i=}")
            print(f"{(county, category)}")
            print(f"{temp_df=}")
            predictions_df = pd.concat([predictions_df, temp_df])

        print("✅ Step 3 completed ")


        # 4. Append predictions to raw data
        df_pred_filter = df_pred[['date_week','county', 'category_name','bottles_sold']].copy()
        df_pred_filter["is_predict"] = False
        predictions_df["is_predict"] = True
        df_combined = pd.concat([df_pred_filter, predictions_df], ignore_index=True)

        print("✅ Step 4 completed ")

        # 5. Sort by date for visualization
        #df_combined = df_combined.sort_values(by=['county', 'category_name', 'date_week'])

        print("✅ Step 5 completed ")

        print("✅  DF Combined Output:")
        print(df_combined.head(10))

        aux = data.query(f"date_week >= '2024-01-01' and date_week <= '2025-01-31' ").copy()
        aux["is_predict"] = False
        aux_real = aux[aux['date_week']>="2025-01-01"]
        y_real = aux_real[['date_week','county', 'category_name','bottles_sold','is_predict']]

        y_combined = pd.concat([y_real, predictions_df], ignore_index=True)
        #y_combined = y_combined.sort_values(by=['county', 'category_name', 'date_week'])

        df_combined = pd.concat([df_combined,y_real], ignore_index=True)
        print("✅  Y Combined Output:")
        print(y_combined.head(10))
        # Store as CSV if the BQ query returned at least one valid line
        if df_combined.shape[0] > 1:
            df_combined.date_week = pd.to_datetime(df_combined.date_week)
            df_combined.to_csv(Path(PRED_DATA_PATH).joinpath("data.csv"), header=True, index=False, date_format='%Y-%m-%d')
            print(f"df_pred creado en {Path(PRED_DATA_PATH).joinpath('data.csv')} ")
        return y_combined


    # print(f"cargando modelo")
    # model = load_model()
    # assert model is not None
    # print(f"modelo cargado")

    # print(f"predicting X_pred")
    # y_pred = model.predict(X_pred)

    # print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    #return y_pred

def prepare_data_to_visualization():
    data = get_data_with_cache(
        gcp_project = GCP_PUBLIC_DATA,
        query = 'hi',
        cache_path=Path(RAW_DATA_PATH).joinpath("data.csv"),
        data_has_header=True
    )
    dummy_data = data.query("date_week >= '2024-01-01' and date_week < '2025-01-01'")
    dummy_data['is_pred'] = False
    dummy_data2 = data.query("date_week >= '2025-01-01'")
    dummy_data2['is_pred'] = True
    dummy_data_df = pd.concat([dummy_data, dummy_data2], axis=0)
    return dummy_data_df

def predict_on_validation(model, X_val, county_mapping, category_mapping):
    print("🔍 Generating validation predictions...")
    y_pred = model.predict(X_val)

    # Convert ID back to names
    county_id_to_name = {v: k for k, v in county_mapping.items()}
    category_id_to_name = {v: k for k, v in category_mapping.items()}

    counties = [county_id_to_name[idx[0]] for idx in X_val["county_id"]]
    categories = [category_id_to_name[idx[0]] for idx in X_val["category_id"]]

    # Create a DataFrame for easy analysis
    df_predictions = pd.DataFrame({
        "county": counties,
        "category": categories,
        "actual": y_val.mean(axis=1),
        "predicted": y_pred.mean(axis=1)
    })

    return df_predictions


import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_predictions(df_predictions, county, category):
    """
    Plot actual vs predicted values for a given county and category.

    Parameters:
    - df_predictions: DataFrame containing "county", "category", "actual", and "predicted".
    - county: The county name to filter.
    - category: The category name to filter.
    """

    # Filter for the selected county and category
    df_filtered = df_predictions[(df_predictions["county"] == county) &
                                 (df_predictions["category"] == category)]

    if df_filtered.empty:
        print(f"No data found for {county} - {category}")
        return

    plt.figure(figsize=(10, 5))

    # Plot actual vs. predicted
    plt.plot(df_filtered.index, df_filtered["actual"], marker='o', label="Actual Sales")
    plt.plot(df_filtered.index, df_filtered["predicted"], linestyle="dashed", marker='x', label="Predicted Sales")

    plt.xlabel("Sample Index")  # No date, just using index
    plt.ylabel("Bottles Sold")
    plt.title(f"Actual vs. Predicted Sales ({county} - {category})")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':

    data = get_data()

    data_processed, county_mapping, category_mapping = preprocess(data)

    # Train Model
    model, X_val, y_val, county_mapping, category_mapping = train(data_processed, county_mapping, category_mapping)

    # Predict on Validation
    df_predictions = predict_on_validation(model, X_val, county_mapping, category_mapping)

    # Plot Results for a specific county & category
    plot_predictions(df_predictions, county="POLK", category="WHISKY")
