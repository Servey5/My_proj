import cloudpickle
import pandas as pd


class FastAPIHandler:

    def __init__(self, model_path="/models/model.pkl"):

        with open(model_path, "rb") as f:
            self.model = cloudpickle.load(f)

    def predict(self, features: dict) -> float:

        # Преобразуем признаки в DataFrame
        df = pd.DataFrame([features])

        # Сохраняем порядок признаков
        if hasattr(self.model, "feature_names_in_"):
            feature_order = self.model.feature_names_in_
            df = df.reindex(columns=feature_order)

        # Получаем предсказание
        prediction = self.model.predict(df)

        return float(prediction[0])