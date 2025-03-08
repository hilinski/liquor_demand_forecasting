import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from liquor_app.interface.main import prepare_data_to_visualization
from arima.amain import prepare_data_to_visualization

from datetime import datetime
import numpy as np
from datetime import datetime
import pytz
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict():   
    data_df, data_mae = prepare_data_to_visualization()
    json_data = data_df.to_json(orient='records')
    json_mae = data_mae.to_json(orient='records')
    return  json_data, json_mae

@app.get('/pred-year')
def pred_year(year_pred: int):
    data_df, data_mae = prepare_data_to_visualization(year=year_pred)
    json_data = data_df.to_json(orient='records')
    json_mae = data_mae.to_json(orient='records') 
    return  json_data, json_mae
    # return { "endpoit " : f'{year_pred}'}
 
    
@app.get("/")
def root():
    return { "hello" : "el team del licor"}

# make a test 