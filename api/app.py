import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from fastapi import FastAPI
from predict import predict_next_8_weeks

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Forecast API Running"
    }

@app.get("/forecast/{state}")
def forecast(state: str):
    preds = predict_next_8_weeks(state)
    return {
        "state": state,
        "forecast": preds
    }
