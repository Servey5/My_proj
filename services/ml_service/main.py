from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
from api_handler import FastAPIHandler

app = FastAPI()

# Для локального запуска путь "../models/model.pkl"
handler = FastAPIHandler(model_path="../models/model.pkl")

@app.get("/")
def root() -> dict:
    return {"Hello": "World"}

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

@app.post("/api/prediction")
def predict(item_id: int, body: PredictionRequest):
    pred_value = handler.predict(body.features)
    return {"item_id": item_id, "price": pred_value}