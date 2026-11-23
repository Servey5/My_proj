import mlflow
import mlflow.sklearn
import cloudpickle
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000")

RUN_ID = "491984e6b2e846f2ab43f5e7ec1bc5b5"

model_uri = f"runs:/{RUN_ID}/model"

print(f"Загрузка модели: {model_uri}")

model = mlflow.sklearn.load_model(model_uri)

# Сохраняем модель в папку, которая подключена к Docker
os.makedirs("../models", exist_ok=True)

with open("../models/model.pkl", "wb") as f:
    cloudpickle.dump(model, f)

print("Модель успешно сохранена!")