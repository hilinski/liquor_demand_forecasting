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
            where date <= '2023-03-31' and date >= '2020-01-01'
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
        case when county in ('POLK','LINN','SCOTT','BLACK HAWK','JOHNSON','POTTAWATTAMIE','DUBUQUE','STORY','WOODBURY','DALLAS') then county else 'OTHER' END AS county,
        case when category_name in ('WHITE RUM','IMPORTED VODKAS','PUERTO RICO & VIRGIN ISLANDS RUM','FLAVORED RUM','100% AGAVE TEQUILA','IMPORTED DRY GINS','SCOTCH WHISKIES','IMPORTED CORDIALS & LIQUEURS','GOLD RUM','SPICED RUM') then category_name else 'OTHER' END AS category_name,
        case when vendor_name in ('SAZERAC COMPANY  INC','DIAGEO AMERICAS','HEAVEN HILL BRANDS','LUXCO INC','JIM BEAM BRANDS','FIFTH GENERATION INC','PERNOD RICARD USA','MCCORMICK DISTILLING CO.','BACARDI USA INC','E & J GALLO WINERY') then vendor_name else 'OTHER' END as vendor_name,
        sum(bottles_sold) as bottles_sold
        FROM clean_data2
        group by 1,2,3,4
        )
        select extract(YEAR FROM date) as year,
        extract(MONTH FROM date) as month,
        extract(DAY FROM date) as day,
        extract(DAYOFWEEK FROM date) as dow,
        extract(WEEK from date) as week,
        *
        from group_and_others
    """

    data = get_data_with_cache(
        gcp_project = GCP_PUBLIC_DATA,
        query = query,
        cache_path=Path(RAW_DATA_PATH).joinpath("data.csv"),
        data_has_header=True
    )

    # # # CLEAN DATA
    # # DATA_CLEAN = CLEAN_DATA(DATA)
    # # PRINT("✅ DATA CLEANED ")

    # # # PROCESS DATA
    # # X = DATA_CLEAN.DROP(['PACK', 'BOTTLE_VOLUME_ML', 'STATE_BOTTLE_COST', 'STATE_BOTTLE_RETAIL',
    # # 'BOTTLES_SOLD', 'SALE_DOLLARS', 'VOLUME_SOLD_LITERS','VOLUME_SOLD_GALLONS'], AXIS=1)
    # # DATES = DATA_CLEAN[['DATE']]
    # # Y = DATA_CLEAN[['BOTTLES_SOLD']]
    # # X_PROCESSED,COL_NAMES = PREPROCESS_FEATURES(X,TRUE)
    # # PRINT("✅ DATA PROCCESED ")

    # # # LOAD A DATAFRAME ONTO BIGQUERY CONTAINING [PICKUP_DATETIME, X_PROCESSED, Y]
    # # # USING DATA.LOAD_DATA_TO_BQ()


    # # X_PROCESSED_DF = PD.DATAFRAME(
    # #     X_PROCESSED,
    # #     COLUMNS=COL_NAMES
    # # )

    # # DATA_PROCESSED = PD.CONCAT([DATES, X_PROCESSED_DF, Y], AXIS="COLUMNS", SORT=FALSE)
    # # #DATA_PROCESSED = PD.DATAFRAME(NP.CONCATENATE((DATES, X_PROCESSED, Y), AXIS=1))
    # # PROCESSED_PATH = PATH(PROCESSED_DATA_PATH).JOINPATH("DATA_PROCESSED.CSV")
    # # DATA_PROCESSED.TO_CSV(PROCESSED_PATH, HEADER=TRUE, INDEX=FALSE)

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
