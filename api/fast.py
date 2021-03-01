from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from predict import get_model
from params import BUCKET_NAME, MODEL_NAME
from TaxiFareModel.trainer import Trainer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.get("/predict_fare")
def predict_fare(pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count):
    # compute `wait_prediction` from `day_of_week` and `time`
    model = get_model('model.joblib')
    params = {
              "key": pd.datetime.now().strftime("%Y/%m/%d, %H:%M:%S"),
              "pickup_datetime": pickup_datetime,
              "pickup_longitude": pickup_longitude,
              "pickup_latitude": pickup_latitude,
              "dropoff_longitude": dropoff_longitude,
              "dropoff_latitude": dropoff_latitude,
              "passenger_count": passenger_count
            }
    X_pred = pd.DataFrame(params, index=[0])
    y_pred = model.predict(X_pred)
    return {"prediction": y_pred[0]}

