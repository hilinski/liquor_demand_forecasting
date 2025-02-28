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
from liquor_app.ml_logic.preprocessor import preprocess_features, create_sequences_padre
from liquor_app.ml_logic.registry import load_model, save_model#, save_results
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

    # # # LOAD A DATAFRAME ONTO BIGQUERY CONTAINING [PICKUP_DATETIME, X_PROCESSED, Y]
    # # # USING DATA.LOAD_DATA_TO_BQ()

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


def train(min_date:str = '2023-01-01',
        max_date:str = '2023-03-31',
        split_ratio: float = 0.20, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
        learning_rate=0.0005,
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

    #tomar solo 10% de la data
    #data = data.sample(frac=0.4, random_state=42)  # Tomar solo el 10% de los datos
    columnas_target = data[["bottles_sold"]].copy()
    columnas_apoyo = data[['category_name','county','sin_MoSold', 'cos_MoSold']].copy()

    data_preproc = data.iloc[:,:-(len(columnas_target.columns)+len(columnas_apoyo.columns))]
    data_preproc = data_preproc.drop('remainder__date_week', axis=1)
    print(f"creando secuencias para modelo RNN...")
    X, y = create_sequences_padre(data_preproc, columnas_target, past_steps=52, future_steps=future_steps)
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

    return val_mae, X_val, y_train, y_val


def evaluate(*args) -> float:
    pass

def pred(X_pred:np.ndarray = None, future_steps= future_steps) -> np.ndarray:

    if X_pred is None:
        print(f"cargando datos dummy para X_pred")
        data = get_data_with_cache(GCP_PUBLIC_DATA,
        query = 'hi',
        cache_path=Path(PROCESSED_DATA_PATH).joinpath("data_processed.csv"),
        data_has_header=True
        )
        # data = data.iloc[-20:, :]
        print(f"{data.shape}")

        columnas_target = data[["bottles_sold"]].copy()
        columnas_apoyo = data[['category_name','county']].copy()
        data_preproc = data.iloc[:,:-(len(columnas_target.columns)+len(columnas_apoyo.columns))]
        X, y = create_sequences_padre(data_preproc, columnas_target, past_steps=52, future_steps=future_steps)
        split_ratio = 0.2
        split_index = int((1-split_ratio) * len(X))
        X_prep = X[split_index:]
        y_prep = y[split_index:]
        print("✅ Pred data created ")
        print("X_prep shape:", X_prep.shape)

        model = load_model()
        assert model is not None
        print(f"modelo cargado")

        print(f"predicting X_pred")
        y_pred = model.predict(X_prep)

        print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
        return y_pred


    print(f"cargando modelo")
    model = load_model()
    assert model is not None
    print(f"modelo cargado")

    print("X_pred shape:", X_pred.shape)
    print(f"predicting X_pred")
    y_pred = model.predict(X_pred)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred

def pred_future(model, last_sequence, future_steps=3):
    pass

if __name__ == '__main__':
    data = get_data()
    preprocess(data)
    val_mae, X_val, y_train, y_val = train()
    #evaluate()
    print(pred(X_val))
