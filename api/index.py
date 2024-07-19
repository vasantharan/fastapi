from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
from sklearn.datasets import load_iris

target_names = load_iris().target_names

app = FastAPI()
rf_model = load('./api/Random_Forest.joblib')
knn_model = load('./api/KNN.joblib')

class input(BaseModel):
    features: list

@app.get("/about")
def test():
    return {"hello world": "hello world"}

@app.post("/predict/rf")
def predict_sklearn(data: input):
    x_input = np.array(data.features).reshape(1, -1)
    prediction = rf_model.predict(x_input)
    return {"prediction": target_names[prediction[0]]}

@app.post("/predict/knn")
def predict_knn(data: input):
    x_input = np.array(data.features).reshape(1, -1)
    prediction = knn_model.predict(x_input)
    return {"prediction": target_names[prediction][0]}
