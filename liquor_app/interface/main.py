import numpy as np
import pandas as pd
import pdb

import os
import sys
sys.path.append(os.path.abspath(".."))

from pathlib import Path
from dateutil.parser import parse

from liquor_app.params import *
from liquor_app.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from liquor_app.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from liquor_app.ml_logic.preprocessor import preprocess_features, create_sequences_padre, create_sequences_inference
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

    data = get_data_with_cache(
        gcp_project = GCP_PUBLIC_DATA,
        query = query,
        cache_path=Path(RAW_DATA_PATH).joinpath("data.csv"),
        data_has_header=True
    )

    return data


def preprocess(data) -> None:
    month_in_year = 12
    data['date_week'] = pd.to_datetime(data['date_week'])
    data['num_month'] = data['date_week'].dt.month
    data['sin_MoSold'] = np.sin(2*np.pi*data.num_month/month_in_year)
    data['cos_MoSold'] = np.cos(2*np.pi*data.num_month/month_in_year)
    data = data.drop('num_month', axis=1)
    columnas_target = data[["bottles_sold"]]
    columnas_apoyo = data[['category_name','county','sin_MoSold','cos_MoSold']]
    data_processed,col_names = preprocess_features(data,True)
    print("✅ Data Proccesed ")

    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # using data.load_data_to_bq()

    data_processed = pd.DataFrame(
        data_processed,
        columns=col_names
    )

    data_processed = pd.concat([data_processed, columnas_apoyo, columnas_target], axis="columns", sort=False)
    data_processed.columns = data_processed.columns.str.replace(" ", "_")
    # data_processed.rename(columns={'remainder__date_ordinal':'date_ordinal'}, inplace=True)
    # #data_processed = pd.DataFrame(np.concatenate((dates, X_processed, y), axis=1))

    processed_path = Path(PROCESSED_DATA_PATH).joinpath("data_processed.csv")
    data_processed.to_csv(processed_path, header=True, index=False)

    print(f"✅ Raw data saved as {RAW_DATA_PATH}")
    print(f"✅ Processed data saved as {PROCESSED_DATA_PATH}")
    print("✅ preprocess() done")
    return data_processed


def train(min_date:str = '2019-01-01',
        max_date:str = '2024-12-31',
        split_ratio: float = 0.083333333, # represents 1 year of the dataset
        learning_rate=0.0105,
        batch_size = 256,
        patience = 10,
        future_steps = future_steps
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print("\n⭐️ Use case: train")
    print( "\nLoading preprocessed validation data...")

    #min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    #max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    # Load processed data using `get_data_with_cache` in chronological order
    # Try it out manually on console.cloud.google.com first!

    query = f"""
        SELECT 'hello world'
        """

    data = get_data_with_cache(GCP_PUBLIC_DATA,
        query,
        cache_path=Path(PROCESSED_DATA_PATH).joinpath("data_processed.csv"),
        data_has_header=True
    )

    data = data.query(f"remainder__date_week >= '{min_date}' and remainder__date_week <= '{max_date}' ")

    #tomar solo 10% de la data
    #data = data.sample(frac=0.4, random_state=42)  # Tomar solo el 10% de los datos
    columnas_target = data[["bottles_sold"]].copy()
    columnas_apoyo = data[['category_name','county','sin_MoSold', 'cos_MoSold']].copy()

    data_preproc = data.iloc[:,:-(len(columnas_target.columns)+len(columnas_apoyo.columns))]
    data_preproc = data_preproc.drop('remainder__date_week', axis=1)
    print(f"creando secuencias para modelo RNN...")
    X, y = create_sequences_padre(data_preproc, columnas_target, past_steps=past_steps, future_steps=future_steps)
    print("✅ Secuencias creadas ")

    split_index = int((1-split_ratio) * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    print("✅ Train/Val Split created ")

    print("Input shape X train completo:", X_train.shape)
    print("Input shape y train completo:", y_train.shape)
    print("Input shape X train[1:]:", X_train.shape[1:])
    print("Input shape X val completo:", X_val.shape)
    print("Input shape y train completo:",y_train.shape)
    print("Input shape y val completo:",y_val.shape)

    print(f"inicializando modelo")
    model = initialize_model(input_shape=X_train.shape[1:], future_steps=future_steps)
    model = compile_model(model, learning_rate=learning_rate)

    print("✅ Model compiled succesfully")

    model,history = train_model(
        model,
        X_train,
        y_train,
        batch_size=batch_size,
        patience=patience,
        validation_data=(X_val, y_val)
    )

    if 'val_mae' in history.history:
        val_mae = np.min(history.history['val_mae'])
    else:
        val_mae = np.min(history.history['val_loss'])  # Use validation loss instead

    print("val_mae:", val_mae)

    # params = dict(
    #     context="train",
    #     row_count=len(X_train),
    # )

    # Save results on the hard drive using taxifare.ml_logic.registry
    #save_results(params=params, metrics=dict(mae=val_mae))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")

    return val_mae, X_val


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

if __name__ == '__main__':
    #data = get_data()
    #preprocess(data)
    train()
    #print(pred(X_val))
    pred(past_steps=past_steps, future_steps=future_steps)
    #data = prepare_data_to_visualization()
    # print(f"{data.shape=}")
    # print(data.head())
    # print(data.tail())
