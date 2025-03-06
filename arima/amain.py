import pandas as pd
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(".."))
from pathlib import Path
from dateutil.parser import parse
import seaborn as sns
import matplotlib.pyplot as plt
# from pivottablejs import pivot_ui
from google.cloud import bigquery
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from google.cloud import bigquery
from arima.aparams import *
import pickle

#ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX

RAW_DATA_ARIMA_PATH = Path(RAW_DATA_PATH).joinpath("data.csv")

def adf_test(series):
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] < 0.05:
        print("✅ The data is stationary.")
    else:
        print("❌ The data is NOT stationary. Differencing needed.")


def get_data(cache_path, min_date='2013-01-01',max_date='2025-01-31'):

    if cache_path.is_file():
        print("\nLoad data from local CSV...")
        df = pd.read_csv(cache_path, header='infer')
    else:
        print("\nLoad data from BigQuery server...")
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
            coalesce(s.bottles_sold,0) as bottles_sold
              from combinations c
              left join summary s on c.date = s.date and c.category_name = s.category_name and c.county = s.county
              )
              select date, category_name,
              sum(bottles_sold) as bottles_sold
               from data_combinations
               group by 1,2
               order by date asc, category_name asc

        """
        gcp_project = GCP_PUBLIC_DATA
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()
        df.to_csv(cache_path, header=True, index=False)

    return df


def preprocess(df):
    df = df.copy()
    df['bottles_sold'] = df['bottles_sold'].fillna(0)
    df.date = pd.to_datetime(df.date)
    df.set_index(['date'], inplace=True)
    df_demand = df.resample("ME")[["bottles_sold"]].sum()
    return df_demand

def train(df_demand,category_name):
##CREAR EL FOR ACA
    #decomposition
    result_add = seasonal_decompose(df_demand["bottles_sold"], model='additive', period= 12)
    m=12

    #train test split
    df_train = df_demand[df_demand.index.year < 2024].copy()
    df_test = df_demand[df_demand.index.year >= 2024].copy()

    #stationarity
    df_demand_diff = df_train.diff().dropna()
    adf_test(df_demand_diff)
    df_demand_diff2 = df_demand_diff.diff().dropna()
    adf_test(df_demand_diff2)

    df_demand_season_diff = df_demand.diff(12).dropna()
    adf_test(df_demand_season_diff)

    d = 2
    #autocorrelation
    q = 1
    #autoregression
    p = 6

    model = SARIMAX(df_demand_diff2, order=(p,0,q), seasonal_order=(1,1,1,m))
    sarima_result = model.fit(maxiter=1000)
    print(f"Model trained for category {category_name}!")

    with open(Path(ARIMA_MODELS_PATH).joinpath(f"{category_name}.pkl"), "wb") as f:
        pickle.dump(sarima_result, f)
        print(f"Model saved in pickle for {category_name}")

    return df_train, df_test


def pred(df_train, df_test, category_name):
    with open(Path(ARIMA_MODELS_PATH).joinpath(f"{category_name}.pkl"), "rb") as f:
        sarima_result = pickle.load(f)
        print(f"Model loaded for {category_name}")

    # Forecast for the next 12 periods on the differenced data
    forecast_diff2 = sarima_result.get_forecast(steps=12)
    forecast_diff2_values = forecast_diff2.predicted_mean

    # Create forecast index for the next 12 months
    forecast_index = pd.date_range(start=df_train.index[-1] + pd.DateOffset(months=1),
                                   periods=12, freq="M")

    # Reverse the second differencing (d=2 → d=1)
    # The forecast_diff2_values should be added to the last value of the first differenced data
    last_diff1 = df_train.diff().dropna().iloc[-1]  # Last value from the first differenced data

    # Apply cumulative sum to forecasted differenced values to reverse second differencing
    forecast_diff1_values = forecast_diff2_values.cumsum() + float(last_diff1)

    # Reverse the first differencing (d=1 → Original Scale)
    # Use the last actual value of the original series to reverse the first differencing
    last_value = df_train.iloc[-1]  # Last actual value of the original data
    forecast_original = forecast_diff1_values.cumsum() + float(last_value)

    # Slice to ensure forecast only contains the last 12 values
    forecast_original = forecast_original[-12:]

    mae = abs(forecast_original.values - df_test.bottles_sold.values[:-1]).mean()
    print(f"MAE for {category_name}: {mae}")

    mre = abs((forecast_original.values - df_test.bottles_sold.values[:-1])/df_test.bottles_sold.values[:-1]).mean()
    print(f"MRE for {category_name}: {mre}")

    df_pred = forecast_original.to_frame().reset_index()
    df_pred.rename(columns={'index':'date','predicted_mean':'bottles_sold'}, inplace=True)
    df_pred['is_pred'] = True
    df_pred.set_index(['date'], inplace=True)
    df_train['is_pred'] = False
    df_consolidado = pd.concat([df_pred,df_train],axis=0)
    df_consolidado = df_consolidado.reset_index()
    df_consolidado['category_name'] = category_name
    return df_consolidado

def prepare_data_to_visualization():
    df = get_data(cache_path = RAW_DATA_ARIMA_PATH)
    df_demand = preprocess(df)
    df_final = pd.DataFrame()
    # category_name = 'RUM'
    for category_name in ['RUM','VODKA','WHISKY','TEQUILA_MEZCAL','LIQUEURS','GIN','OTROS']:
        print(f"Empezando entrenamiento de {category_name}")
        df_train, df_test = train(df_demand,category_name)
        df_consolidado = pred(df_train, df_test, category_name)
        df_final = pd.concat([df_final, df_consolidado], axis=0)
        
    print(df_final)
    
    return df_final
        
if __name__ == '__main__':
    df = get_data(cache_path = RAW_DATA_ARIMA_PATH)
    df_demand = preprocess(df)
    df_final = pd.DataFrame()
    for category_name in ['RUM','VODKA','WHISKY','TEQUILA_MEZCAL','LIQUEURS','GIN','OTROS']:
        print(f"Empezando entrenamiento de {category_name}")
        df_train, df_test = train(df_demand,category_name)
        df_consolidado = pred(df_train, df_test, category_name)
        df_final = pd.concat([df_final, df_consolidado], axis=0)
    print(df_final)
