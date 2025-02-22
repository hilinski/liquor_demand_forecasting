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
from liquor_app.ml_logic.preprocessor import preprocess_features, crear_secuencias
from liquor_app.ml_logic.registry import load_model, save_model#, save_results

def preprocess(min_date='2013-01-01', max_date='2023-06-30', *args) -> None:
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

    data_clean = get_data_with_cache(
        gcp_project = GCP_PUBLIC_DATA,
        query = query,
        cache_path=Path(RAW_DATA_PATH).joinpath("data.csv"),
        data_has_header=True
    )

    # # # CLEAN DATA
    # # DATA_CLEAN = CLEAN_DATA(DATA)
    # # PRINT("✅ DATA CLEANED ")

    # Process data
    X = data_clean.drop(['pack', 'bottle_volume_ml', 'state_bottle_cost', 'state_bottle_retail',
                         'sale_dollars', 'volume_sold_liters','volume_sold_gallons'], axis=1, errors='ignore')
    y = data_clean[['bottles_sold']]
    X_processed,col_names = preprocess_features(X,True)
    print("✅ Data Proccesed ")

    # # # LOAD A DATAFRAME ONTO BIGQUERY CONTAINING [PICKUP_DATETIME, X_PROCESSED, Y]
    # # # USING DATA.LOAD_DATA_TO_BQ()


    # # X_PROCESSED_DF = PD.DATAFRAME(
    # #     X_PROCESSED,
    # #     COLUMNS=COL_NAMES
    # # )

    data_processed = pd.concat([X_processed_df, y], axis="columns", sort=False)
    data_processed.rename(columns={'remainder__date_ordinal':'date_ordinal'}, inplace=True)
    #data_processed = pd.DataFrame(np.concatenate((dates, X_processed, y), axis=1))
    processed_path = Path(PROCESSED_DATA_PATH).joinpath("data_processed.csv")
    data_processed.to_csv(processed_path, header=True, index=False)

    print(f"✅ Raw data saved as {RAW_DATA_PATH}")
    print(f"✅ Processed data saved as {PROCESSED_DATA_PATH}")
    print("✅ preprocess() done")
    return data_processed


def train(min_date:str = '2023-01-01',
        max_date:str = '2023-03-31',
        split_ratio: float = 0.10, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
        learning_rate=0.0005,
        batch_size = 256,
        patience = 5
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print("\n⭐️ Use case: train")
    print( "\nLoading preprocessed validation data...")

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

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
    data = data.sample(frac=0.1, random_state=42)  # Tomar solo el 10% de los datos

    # Create (X_train_processed, y_train, X_val_processed, y_val)
    train_length = int(len(data) * (1 - split_ratio))

    data_train = data.iloc[:train_length, :].sample(frac=1)
    data_val = data.iloc[train_length:, :].sample(frac=1)

    X_train = data_train.drop(['bottles_sold'], axis=1)
    y_train = data_train[["bottles_sold"]]

    X_val = data_val.drop(['bottles_sold'], axis=1)
    y_val = data_val[['bottles_sold']]

    # Create (X_train_processed, X_val_processed) using `preprocessor.py`
    # Luckily, our preprocessor is stateless: we can `fit_transform` both X_train and X_val without data leakage!
    print(f"{X_train.shape=}")
    print(f"{X_train.shape[1:]=}")
    # Train model using `model.py`

    print(f"creando secuencias train para modelo RNN...")
    X_train_rnn, y_train_rnn = crear_secuencias(X_train, y_train, pasos=10)
    print(f"creando secuencias val para modelo RNN...")
    X_val_rnn, y_val_rnn = crear_secuencias(X_val, y_val, pasos=10)

    print(f"inicializando modelo")
    print("Shape Train RNN con secuencia general:", X_train_rnn.shape)
    print("Shape Train RNN con secuencia filtrado:", X_train_rnn.shape[1:])

    model = initialize_model(input_shape=X_train_rnn.shape[1:])
    model = compile_model(model, learning_rate=learning_rate)

    model,history = train_model(
        model,
        X_train_rnn,
        y_train_rnn,
        batch_size=batch_size,
        patience=patience,
        validation_data=(X_val_rnn, y_val_rnn)
    )

    if 'val_mae' in history.history:
        val_mae = np.min(history.history['val_mae'])
    else:
        val_mae = np.min(history.history['val_loss'])  # Use validation loss instead

    print("val_mae:", val_mae)

    params = dict(
        context="train",
        row_count=len(X_train),
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    #save_results(params=params, metrics=dict(mae=val_mae))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")

    return val_mae


def evaluate(*args) -> float:
    pass

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:

    if X_pred is None:
        print(f"cargando datos dummy para X_pred")
        data = get_data_with_cache(GCP_PUBLIC_DATA,
        query = 'hi',
        cache_path=Path(PROCESSED_DATA_PATH).joinpath("data_processed.csv"),
        data_has_header=True
        )
        data = data.iloc[-20:, :]
        print(f"{data.shape}")
        X_pred = data.drop(['bottles_sold'], axis=1)
        X_pred, X_pred2 = crear_secuencias(X_pred, X_pred, pasos=10)
        print(f"{X_pred}")

    print(f"cargando modelo")
    model = load_model()
    assert model is not None
    print(f"modelo cargado")

    print(f"predicting X_pred")
    y_pred = model.predict(X_pred)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred

if __name__ == '__main__':
    preprocess()
    #train()
    #evaluate()
    #pred()
