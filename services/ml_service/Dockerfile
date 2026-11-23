FROM python:3.11-slim

# Копируем содержимое текущей директории в /ml_service в контейнере
COPY . /ml_service

# Делаем директорию /models доступной снаружи контейнера (volume)
VOLUME /models

# Устанавливаем рабочую директорию
WORKDIR /ml_service

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт 8000
EXPOSE 8000

# Финальная команда запуска сервера
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- КОМАНДЫ ДЛЯ ЗАПУСКА (Пункты 15 и 16) ---
# docker build -t game_sales_service:1 .
# docker run -d --name game_sales_service -p 8000:8000 -v $(pwd)/../models:/models game_sales_service:1