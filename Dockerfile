FROM python:3.8.6-buster

COPY model.joblib /model.joblib
COPY TaxiFareModel /TaxiFareModel
COPY predict.py /predict.py
COPY api /api
COPY params.py /params.py
COPY requirements.txt /requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
