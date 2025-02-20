import numpy as np
import pandas as pd
import pdb

from pathlib import Path
from dateutil.parser import parse

from liquor_app.params import *
from liquor_app.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from liquor_app.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from liquor_app.ml_logic.preprocessor import preprocess_features
from liquor_app.ml_logic.registry import load_model, save_model#, save_results

def preprocess(*args) -> None:
    query = """
        with clean_data as (
            select * EXCEPT (store_number, zip_code, category, vendor_number, county_number),
            CAST(store_number AS NUMERIC) as store_number ,
            CAST(zip_code AS NUMERIC) as zip_code ,
            CAST(category AS NUMERIC) as category,
            CAST(vendor_number AS NUMERIC) as vendor_number
            from `bigquery-public-data.iowa_liquor_sales.sales`
            where date <= '2023-03-31' and date >= '2023-01-01'
            and CAST(vendor_number AS NUMERIC) in (260,421,65,370,85,434,35,301,259,115,395,55,420,205,380,192,297,300,255,389)
            ORDER BY date ASC
        ),
        distinct_vendor as (
            select
                CAST(vendor_number AS NUMERIC) as vendor_number,
                ARRAY_AGG(vendor_name ORDER BY date DESC LIMIT 1) as vendor_name
            from `bigquery-public-data.iowa_liquor_sales.sales`
            group by 1
        ),
        distinct_category as (
            select
                CAST(category AS NUMERIC) as category,
                ARRAY_AGG(category_name ORDER BY date DESC LIMIT 1) as category_name
            from `bigquery-public-data.iowa_liquor_sales.sales`
            group by 1
        ),
        distinct_store as (
            select
                CAST(store_number AS NUMERIC) as store_number,
                ARRAY_AGG(store_name ORDER BY date DESC LIMIT 1) as store_name
            from `bigquery-public-data.iowa_liquor_sales.sales`
            group by 1
        )
        select
            cd.* EXCEPT (vendor_name, category_name, store_name),
            dv.vendor_name,
            dc.category_name,
            ds.store_name
        from clean_data cd
        left join distinct_vendor dv on cd.vendor_number = dv.vendor_number
        left join distinct_category dc on cd.category = dc.category
        left join distinct_store ds on cd.store_number = ds.store_number
    """

    data = get_data_with_cache(
        gcp_project = GCP_PUBLIC_DATA,
        query = query,
        cache_path=Path(RAW_DATA_PATH).joinpath("data.csv"),
        data_has_header=True
    )

    # Clean data
    data_clean = clean_data(data)
    print("✅ Data cleaned ")

    # Process data
    X = data_clean.drop(['pack', 'bottle_volume_ml', 'state_bottle_cost', 'state_bottle_retail',
    'bottles_sold', 'sale_dollars', 'volume_sold_liters','volume_sold_gallons'], axis=1)
    #dates = data_clean[['date']]
    y = data_clean[['bottles_sold']]
    X_processed,col_names = preprocess_features(X,True)
    print("✅ Data Proccesed ")

    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # using data.load_data_to_bq()


    X_processed_df = pd.DataFrame(
        X_processed,
        columns=col_names
    )

    data_processed = pd.concat([X_processed_df, y], axis="columns", sort=False)
    data_processed.rename(columns={'remainder__date_ordinal':'date_ordinal'}, inplace=True)
    #data_processed = pd.DataFrame(np.concatenate((dates, X_processed, y), axis=1))
    processed_path = Path(PROCESSED_DATA_PATH).joinpath("data_processed.csv")
    data_processed.to_csv(processed_path, header=True, index=False)

    print(f"✅ Raw data saved as {RAW_DATA_PATH}")
    print(f"✅ Processed data saved as {PROCESSED_DATA_PATH}")
    print("✅ preprocess() done")


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

    #X_train = data_train.drop(["fare_amount","pickup_datetime"], axis=1)
    #y_train = data_train[["fare_amount"]]

    X_train = data_train.drop(['bottles_sold'], axis=1)
    y_train = data_train[["bottles_sold"]]

    #X_val = data_val.drop(["fare_amount","pickup_datetime"], axis=1)
    #y_val = data_val[["fare_amount"]]

    X_val = data_val.drop(['bottles_sold'], axis=1)
    y_val = data_val[['bottles_sold']]

    # Create (X_train_processed, X_val_processed) using `preprocessor.py`
    # Luckily, our preprocessor is stateless: we can `fit_transform` both X_train and X_val without data leakage!
    print(f"{X_train.shape=}")
    print(f"{X_train.shape[1:]=}")
    # Train model using `model.py`

    # crear secuencias de RNN
    def crear_secuencias(X, y, pasos=10):
        X, y = np.array(X), np.array(y)  # Convertir a arrays NumPy si aún no lo son
        secuencias_X = np.array([X[i:i+pasos] for i in range(len(X) - pasos)])
        secuencias_y = np.array([y[i+pasos] for i in range(len(y) - pasos)])

        return np.array(secuencias_X), np.array(secuencias_y)

    print(f"creando secuencias train para modelo RNN...")
    X_train_rnn, y_train_rnn = crear_secuencias(X_train, y_train, pasos=10)
    print(f"creando secuencias val para modelo RNN...")
    X_val_rnn, y_val_rnn = crear_secuencias(X_val, y_val, pasos=10)

    print(f"inicializando modelo")
    model = initialize_model(input_shape=X_train_rnn.shape[1:])
    model = compile_model(model, learning_rate=learning_rate)

    model, history = train_model(
        model, X_train_rnn, y_train_rnn,
        batch_size=batch_size,
        patience=patience,
        validation_data=(X_val_rnn, y_val_rnn)
    )

    val_mae = np.min(history.history['val_mae'])

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
    pass

if __name__ == '__main__':
    #preprocess()
    train()
    #evaluate()
    #pred()
