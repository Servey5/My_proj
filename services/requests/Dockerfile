FROM python:3.11-slim

# Создаём рабочую директорию /app внутри контейнера и переходим в неё.
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY send_requests.py .

# отключает буферизацию стандартного вывода Python.
ENV PYTHONUNBUFFERED=1

CMD ["python", "send_requests.py"]

# Сборка образа из каталога services/requests:
# docker build -t game_requests:1 .

# Пример запуска из каталога services:
# docker run --rm --network services_default -e API_URL=http://ml_service:8000/api/prediction game_requests:1
