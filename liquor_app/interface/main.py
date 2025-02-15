import numpy as np
import pandas as pd
import pdb

from pathlib import Path

from liquor_app.params import *
from liquor_app.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from liquor_app.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from liquor_app.ml_logic.preprocessor import preprocess_features
from liquor_app.ml_logic.registry import load_model, save_model, save_results

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
    dates = data_clean[['date']]
    y = data_clean[['bottles_sold']]
    X_processed,col_names = preprocess_features(X,True)
    print("✅ Data Proccesed ")

    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # using data.load_data_to_bq()


    X_processed_df = pd.DataFrame(
        X_processed,
        columns=col_names
    )

    data_processed = pd.concat([dates, X_processed_df, y], axis="columns", sort=False)
    #data_processed = pd.DataFrame(np.concatenate((dates, X_processed, y), axis=1))
    processed_path = Path(PROCESSED_DATA_PATH).joinpath("data_processed.csv")
    data_processed.to_csv(processed_path, header=True, index=False)

    print(f"✅ Raw data saved as {RAW_DATA_PATH}")
    print(f"✅ Processed data saved as {PROCESSED_DATA_PATH}")
    print("✅ preprocess() done")


def train(*args) -> float:
    pass

def evaluate(*args) -> float:
    pass

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    pass

if __name__ == '__main__':
    preprocess()
    #train()
    #evaluate()
    #pred()
