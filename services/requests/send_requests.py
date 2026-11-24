import os
import random
import time

import requests


API_URL = os.getenv(
    "API_URL",
    "http://ml_service:8000/api/prediction"
)

features = {
    "platform": "PC",
    "platform_type": "PC",
    "platform_maker": "Various",
    "platform_generation": 0,
    "genre": "Action",
    "year": 2020,
    "publisher": "Unknown",
    "developer": "Unknown",
    "publisher_region": "NA",
    "publisher_tier": "AA",
    "esrb_rating": "M",
    "metacritic_score": 85,
    "user_score": 8.5,
    "critic_review_count": 100,
    "user_review_count": 500,
    "launch_price_usd": 60,
    "is_sequel": 0,
    "online_multiplayer": 1,
    "dlc_released": 0,
    "microtransactions": 0,
    "loot_boxes": 0,
    "game_pass_available": 0,
    "vr_support": 0,
    "goty_nominated": 0,
    "goty_won": 0,
    "how_long_to_beat_main_hrs": 20,
    "how_long_to_beat_completionist_hrs": 40,
    "score_group": "High"
}


def send_request():
    item_id = random.randint(1, 100000)

    current_features = features.copy()

    current_features["metacritic_score"] = random.randint(40, 100)
    current_features["user_score"] = round(random.uniform(4, 10), 1)
    current_features["launch_price_usd"] = random.choice(
        [20, 30, 40, 50, 60, 70]
    )

    payload = {   #Формируем тело HTTP-запроса.
        "features": current_features
    }

    try:
        response = requests.post(
            API_URL,
            params={"item_id": item_id},
            json=payload,
            timeout=30
        )

        print(
            f"item_id={item_id} | "
            f"status={response.status_code} | "
            f"response={response.text}",
            flush=True
        )

    except requests.exceptions.RequestException as error:
        print(
            f"Ошибка отправки запроса: {error}",
            flush=True
        )


if __name__ == "__main__":
    print(f"Сервис запросов запущен: {API_URL}", flush=True)

    while True:
        send_request()

        delay = random.uniform(0, 5)

        print(
            f"Следующий запрос через {delay:.2f} сек.",
            flush=True
        )

        time.sleep(delay)
