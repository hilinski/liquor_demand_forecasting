import pandas as pd

from google.cloud import bigquery
from pathlib import Path

from liquor_app.params import *

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    #Get DataFrame cleaned from NaN values, negative values in sold columns
    # and drop "county_number" empty column
    assert isinstance(df, pd.DataFrame)
    df = df.drop(['invoice_and_item_number','store_name','store_number','vendor_number','category','address','city','zip_code','item_number','item_description','store_location'], axis=1, errors='ignore')
    df[df['bottles_sold']>=0]
    return df

def get_data_with_cache(
        gcp_project:str,
        query:str,
        cache_path:Path,
        data_has_header=True
    ) -> pd.DataFrame:
    """
    Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
    Store at `cache_path` if retrieved from BigQuery for future use
    """
    print(cache_path)
    if cache_path.is_file():
        print("\nLoad data from local CSV...")
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
    else:
        print("\nLoad data from BigQuery server...")
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()


        # Store as CSV if the BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df

def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(f"\nSave data to BigQuery @ {full_table_name}...:")

    # Load data onto full_table_name

    # 🎯 HINT for "*** TypeError: expected bytes, int found":
    # After preprocessing the data, your original column names are gone (print it to check),
    # so ensure that your column names are *strings* that start with either
    # a *letter* or an *underscore*, as BQ does not accept anything else
    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

    client = bigquery.Client()

    if truncate:
        write_mode = "WRITE_TRUNCATE"
    else:
        write_mode = "WRITE_APPEND"

    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()
    print(result)

    print(f"✅ Data saved to bigquery, with shape {data.shape}")
