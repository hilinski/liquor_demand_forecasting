import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from datetime import datetime
import numpy as np
from datetime import datetime
import pytz


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
    
    return {"fare"}

@app.get("/")
def root():
    return {"greeting": "Hello"}