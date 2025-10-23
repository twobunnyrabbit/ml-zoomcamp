import pickle
from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


record_1 = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
}

record_2 = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
}

with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

app = FastAPI(title="converted")


def single_predict(client):
    result = pipeline.predict_proba([client])[0, 1]
    print(result)
    return float(result)


@app.post("/predict")
def predict(client: Item):
    print("app.post()..client")
    print(client)
    # Convert Pydantic model to dict for sklearn pipeline
    client_dict = client.model_dump()
    print("client_dict...")
    print(client_dict)
    convert = single_predict(client_dict)
    return convert


# Question 1
"""
Question 1
uv version 0.9.5
"""

# Queston 2
"""
Question 2

What's the first hash for scikit-learn

ha256:b4fc2525eca2c69a59260f583c56a7557c6ccdf8deafdba6e060f94c1c59738e
"""

# Question 3
"""
Question 3

What's the probability that this lead will convert?

probability this lead will convert is 0.533
"""

# Question 4
"""
Question 4

Serving model as a web service

uvicorn homework5-deployment:app --host "0.0.0.0" --port "9696" --reload

What's the probability that this client will get a subscription?
0.534

"""

# Question 5
"""
Question 5

docker pull agrigorev/zoomcamp-model:2025

base image size is 121 MB

docker build -t ml-zoomcamp-homework5 .

docker build --platform linux/amd64 -t ml-zoomcamp-homework5 .

docker  run -it --rm -p 9696:9696 ml-zoomcamp-homework5
"""

# Question 6
"""
Question 6

url = "http://localhost:9696/predict"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
requests.post(url, json=client).json()

got 0.5340417283801275
closest answer is 0.59
"""
