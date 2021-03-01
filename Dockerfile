FROM python:3.8.6-buster

COPY model.joblib /model.joblib
COPY TaxiFareModel /TaxiFareModel
COPY predict.py /predict.py
COPY api /api
COPY params.py /params.py
COPY requirements.txt /requirements.txt
COPY /Users/david/Dokumente/gcp_keys/new_9c70605337ae.json /credentials.json

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
