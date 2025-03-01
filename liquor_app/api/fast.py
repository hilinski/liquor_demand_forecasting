import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from liquor_app.interface.main import prepare_data_to_visualization

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
    data = prepare_data_to_visualization()

    json_data = data.to_json(orient='records')
    return  json_data 

@app.get("/")
def root():
    return { "hello" : "el team del licor"}