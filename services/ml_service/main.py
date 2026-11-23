import time
from typing import Any, Dict

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from api_handler import FastAPIHandler


# --------------------------------------------------
# 1. Создание FastAPI-приложения
# --------------------------------------------------

app = FastAPI(
    title="Game Sales Prediction Service",
    description="ML prediction service with Prometheus monitoring",
    version="2.0"
)


# --------------------------------------------------
# 2. Загрузка обученной модели
# --------------------------------------------------

handler = FastAPIHandler(
    model_path="/models/model.pkl"
)


# --------------------------------------------------
# 3. Метрика предсказаний модели
# --------------------------------------------------

PREDICTION_HISTOGRAM = Histogram(
    "model_prediction_value",
    "Distribution of ML model predictions",
    buckets=(
        0,
        5,
        10,
        15,
        20,
        30,
        50,
        100,
        200,
        500,
        1000,
        5000,
        float("inf")
    )
)


# --------------------------------------------------
# 4. Количество HTTP-запросов
# --------------------------------------------------

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"]
)


# --------------------------------------------------
# 5. Время обработки HTTP-запросов
# --------------------------------------------------

HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "Duration of HTTP requests",
    ["method", "endpoint"]
)


# --------------------------------------------------
# 6. Middleware для автоматического сбора метрик
# --------------------------------------------------

@app.middleware("http")
async def collect_metrics(request: Request, call_next):

    start_time = time.perf_counter()

    method = request.method
    endpoint = request.url.path

    status_code = 500

    try:
        response = await call_next(request)

        status_code = response.status_code

        return response

    finally:
        duration = time.perf_counter() - start_time

        if endpoint != "/metrics":

            HTTP_REQUESTS.labels(
                method=method,
                endpoint=endpoint,
                status=str(status_code)
            ).inc()

            HTTP_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)


# --------------------------------------------------
# 7. Проверка работы API
# --------------------------------------------------

@app.get("/")
def root():
    return {"Hello": "World"}


# --------------------------------------------------
# 8. Структура входного JSON
# --------------------------------------------------

class PredictionRequest(BaseModel):
    features: Dict[str, Any]


# --------------------------------------------------
# 9. Endpoint предсказания
# --------------------------------------------------

@app.post("/api/prediction")
def predict(item_id: int, body: PredictionRequest):

    pred_value = handler.predict(body.features)

    pred_value = float(pred_value)

    PREDICTION_HISTOGRAM.observe(pred_value)

    return {
        "item_id": item_id,
        "price": pred_value
    }


# --------------------------------------------------
# 10. Endpoint для Prometheus
# --------------------------------------------------

@app.get("/metrics", include_in_schema=False)
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# --------------------------------------------------
# 11. Учебный endpoint для проверки ошибок 5xx
# --------------------------------------------------

@app.get("/test-error", include_in_schema=False)
def test_error():

    return Response(
        content='{"detail":"Test error for monitoring"}',
        status_code=500,
        media_type="application/json"
    )
