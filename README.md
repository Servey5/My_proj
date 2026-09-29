# Video Game Sales & Metacritic Intelligence

## Описание проекта

Проект посвящён анализу датасета видеоигр (50 000 записей, 1985–2026) и построению
модели машинного обучения для задачи **регрессии** — предсказания мировых продаж игры.

- **Целевая переменная:** `global_sales_million` — мировые продажи в млн копий.
- **Признаки:** платформа, жанр, издатель, разработчик, оценки Metacritic
  (критиков и пользователей), год выпуска, цена запуска, бинарные признаки
  (сиквел, онлайн-режим, DLC, микротранзакции и др.).
- **Ограничение:** данные синтетические — встречаются несоответствия
  (например, «eFootball 2023» на Game Boy 1990 года). Это нужно учитывать
  при интерпретации результатов.

Проект состоит из двух лабораторных работ:

- **ЛР1** — разведочный анализ данных (EDA), очистка датасета.
- **ЛР2** — обучение модели, feature engineering, подбор гиперпараметров,
  логирование экспериментов в MLflow.

## Запуск

### 1. Клонирование репозитория

```bash
git clone git@github.com:username/my_proj.git
cd my_proj
```

### 2. Создание виртуального окружения

```bash
python3 -m venv .venv_my_proj
```

### 3. Активация виртуального окружения

```bash
source .venv_my_proj/bin/activate
```

### 4. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 5. Подготовка данных

Положить исходный файл `games.csv` в папку `data/`:

```
data/
 └── games.csv
```

> Файлы данных **не коммитятся** в репозиторий (см. `.gitignore`) —
> их нужно скачать отдельно и положить в `data/` вручную.

### 6. Запуск EDA (ЛР1)

Открыть `eda/eda.ipynb` в VS Code (или Jupyter) и выполнить все ячейки
сверху вниз. В ходе выполнения создадутся:

- `eda/01_target_distribution.png` … `eda/05_corr_heatmap.png` — статические графики
- `eda/06_interactive_scores.html` — интерактивный график
- `data/clean_dataset.pkl` — очищенный датасет (для ЛР2). **Не коммитится**.

### 7. Запуск MLflow (ЛР2)

MLflow запускается локально, backend store — SQLite, artifact store — локальная папка.

```bash
cd mlflow
sh start_mlflow.sh
```

Скрипт `mlflow/start_mlflow.sh`:

```bash
#!/bin/bash
mlflow server \
    --backend-store-uri sqlite:///mlruns.db \
    --default-artifact-root ./mlartifacts \
    --host 127.0.0.1 \
    --port 5000
```

**UI MLflow:** <http://127.0.0.1:5000/>

> ⚠️ Скрипт `start_mlflow.sh` **должен быть закоммичен**.
> В `.gitignore` добавлены `mlflow/mlartifacts/` и `mlflow/mlruns.db`.

### 8. Запуск исследования модели (ЛР2)

Открыть `research/research.ipynb` и выполнить все ячейки сверху вниз.
Ноутбук воспроизводит все эксперименты:

1. Загрузка `data/clean_dataset.pkl`, split 75/25.
2. Baseline-модель + логирование в MLflow.
3. Feature extraction (`PolynomialFeatures` + `KBinsDiscretizer`) + логирование.
4. Отбор N важнейших признаков по `feature_importances_` + логирование.
5. Подбор гиперпараметров через Optuna + логирование.
6. Обучение финальной модели на полной выборке + регистрация как `Production`.

## Зависимости

- `pandas` — обработка табличных данных
- `numpy` — численные операции
- `matplotlib`, `seaborn` — статические графики
- `plotly` — интерактивные графики
- `scikit-learn` — модели, препроцессинг, метрики
- `mlflow` — трекинг экспериментов и Model Registry
- `sqlalchemy`, `alembic` — backend store для MLflow
- `optuna` — подбор гиперпараметров

### Зафиксированные версии (ЛР2)

```
numpy==1.26.4
scikit-learn==1.5.2
scipy==1.13.1
mlflow==2.16.0
sqlalchemy==1.4.52
alembic==1.13.2
optuna==3.6.1
```

## Структура проекта

```
my_proj
 |_____ .venv_my_proj/           # виртуальное окружение (не коммитится)
 |_____ .git/                    # git-репозиторий
 |_____ data/
 |        |___ games.csv              # исходные данные (не коммитится)
 |        |___ clean_dataset.pkl      # очищенный датасет (не коммитится)
 |
 |_____ eda/
 |        |___ eda.ipynb                          # блокнот с EDA
 |        |___ 01_target_distribution.png
 |        |___ 02_sales_by_platform_type.png
 |        |___ 03_sales_by_year.png
 |        |___ 04_metacritic_vs_sales.png
 |        |___ 05_corr_heatmap.png
 |        |___ 06_interactive_scores.html
 |
 |_____ research/
 |        |___ research.ipynb                     # блокнот с экспериментами
 |        |___ fe_sklearn_columns.csv             # список FE-признаков
 |        |___ selected_features_idx.csv          # индексы отобранных признаков
 |        |___ selected_features_names.csv        # имена отобранных признаков
 |        |___ runs_summary.csv                   # сводка по прогонам MLflow
 |        |___ model_runs.png                     # скриншот прогонов
 |        |___ model_versions.png                 # скриншот версий модели
 |        |___ MLModel                            # артефакт Production-модели
 |
 |_____ mlflow/
 |        |___ start_mlflow.sh                    # скрипт запуска MLflow
 |        |___ mlruns.db                          # backend store (не коммитится)
 |        |___ mlartifacts/                       # artifact store (не коммитится)
 |
 |_____ .gitignore
 |_____ README.md
 |_____ requirements.txt
```

## Описание модулей проекта

### `eda/eda.ipynb`

Блокнот с полным разведочным анализом данных. Основные разделы:

1. **Загрузка данных и знакомство** — описание признаков, статистики,
   приведение типов.
2. **Очистка данных** — проверка пропусков/дубликатов/диапазонов,
   отсечение невалидных строк.
3. **Анализ признаков** — 6 графиков с выводами после каждого.
4. **Сохранение финального датасета** в `data/clean_dataset.pkl`.

### `research/research.ipynb`

Блокнот с экспериментами по построению и настройке модели. Основные разделы:

1. Загрузка `data/clean_dataset.pkl` и split 75/25.
2. Baseline-модель (`RandomForest`) + логирование в MLflow.
3. Feature extraction (`PolynomialFeatures`, `KBinsDiscretizer`) + логирование.
4. Отбор важных признаков по `feature_importances_` (топ-40 %) + логирование.
5. Подбор гиперпараметров через **Optuna** (10 trials, минимизация MAE).
6. Обучение лучшей модели на полной выборке, регистрация в Model Registry
   с тегом `Production`.

### `mlflow/start_mlflow.sh`

Скрипт запуска MLflow-сервера с backend store в SQLite и artifact store
в локальной директории. Запускается из директории `mlflow/`.

---

## Результаты и выводы EDA (ЛР1)

**Очистка данных:**

- Пропусков, дубликатов и выбросов — нет.
- Удалено **3 968 строк (7.94 %)** — free-to-play игры
  (`launch_price_usd == 0`), так как их модель монетизации
  несовместима с задачей предсказания продаж премиум-игр.
- Итоговый размер: **46 032 строки × 33 столбца**.
- Типы данных приведены к компактным — потребление памяти снижено
  с 13.4 МБ до 4.0 МБ.
- Удалены идентификаторы `game_id` и `title`.
- Столбцы `na/eu/jp/other_sales_million` и `estimated_revenue_million_usd`
  **исключены перед обучением модели** (ЛР2) — из-за утечки данных.

**Новые признаки:** на этапе EDA не создавались. Рекомендовано добавить
в ЛР2: `log_global_sales`, `critic_minus_user`, `game_age`, агрегаты по
издателю/платформе.

**Ключевые закономерности:**

1. Целевая переменная сильно скошена (skew ≈ 7.74), после `log1p` — 0.17.
   Для регрессии нужны логарифмирование или устойчивые метрики.
2. Тип платформы слабо разделяет данные — нужны взаимодействия с жанром/годом.
3. Год выпуска несёт временной тренд — обязательный признак.
4. Оценки критиков и игроков — умеренные предикторы
   (`corr` ≈ 0.315 и 0.229 соответственно). Использовать раздельно.
5. Ни один признак не доминирует → в ЛР2 нужна **нелинейная модель**
   (градиентный бустинг, случайный лес).

**Топ-10 признаков по |corr| с целевой:**

| #  | Признак              | \|corr\| |
|----|----------------------|----------|
| 1  | `metacritic_score`   | 0.225    |
| 2  | `user_score`         | 0.159    |
| 3  | `is_sequel`          | 0.143    |
| 4  | `year`               | 0.142    |
| 5  | `launch_price_usd`   | 0.140    |
| 6  | `platform_generation`| 0.106    |
| 7  | `goty_nominated`     | 0.105    |
| 8  | `online_multiplayer` | 0.097    |
| 9  | `loot_boxes`         | 0.089    |
| 10 | `microtransactions`  | 0.083    |

---

## Результаты исследования (ЛР2)

### Схема экспериментов

| # | Эксперимент          | Run name                    | Что добавлено                                                        |
|---|----------------------|-----------------------------|----------------------------------------------------------------------|
| 1 | Baseline             | `baseline_random_forest`    | `StandardScaler` + `OrdinalEncoder` + RF                             |
| 2 | FE через sklearn     | `fe_sklearn_random_forest`  | + `PolynomialFeatures(deg=2)`, + `KBinsDiscretizer`                  |
| 3 | Отбор признаков      | `fe_sklearn_selected_rf`    | + селектор по топ-40 % важности RF                                   |
| 4 | Optuna               | `optuna_best_rf`            | подбор `n_estimators` / `max_depth` / `max_features`                 |
| 5 | Production           | `PRODUCTION_full_data`      | обучение на полной выборке                                           |

### Сводка по всем прогонам эксперимента `game_sales_regression`

| Run name                    | MAE         | MAPE    | MSE      |
|-----------------------------|-------------|---------|----------|
| `optuna_best_rf`            | **15.1938** | 1.0920  | 1107.44  |
| `fe_sklearn_selected_rf`    | 15.2515     | 1.0884  | 1105.75  |
| `baseline_random_forest`    | 15.3642     | 1.1149  | 1119.89  |
| `fe_sklearn_random_forest`  | 15.3649     | 1.1156  | 1115.50  |
| `PRODUCTION_full_data`      | —           | —       | —        |

### Лучшая модель

**`RandomForestRegressor`** с гиперпараметрами, подобранными Optuna:

- `n_estimators = 100`
- `max_depth = 12`
- `max_features = 0.6507`

**Метрики на тестовой выборке:**

| Метрика | Значение |
|---------|----------|
| MAE     | 15.1938  |
| MAPE    | 1.0920   |
| MSE     | 1107.44  |

**Run ID:** `f2437d80947b4e3ebfc71eaa455decec`

### Отобранные признаки

Топ-13 по важности RandomForest (≈ 40 % от 34 FE-признаков).
Полный список — `research/selected_features_names.csv`.

**Топ-10 важных признаков:**

| #  | Признак                                   | Важность |
|----|-------------------------------------------|----------|
| 1  | `cat__publisher_tier`                     | 0.1820   |
| 2  | `num__launch_price_usd`                   | 0.1606   |
| 3  | `cat__genre`                              | 0.0714   |
| 4  | `num__how_long_to_beat_main_hrs`          | 0.0590   |
| 5  | `poly__metacritic_score`                  | 0.0490   |
| 6  | `poly__metacritic_score^2`                | 0.0436   |
| 7  | `num__year`                               | 0.0418   |
| 8  | `num__is_sequel`                          | 0.0410   |
| 9  | `num__metacritic_score`                   | 0.0385   |
| 10 | `num__how_long_to_beat_completionist_hrs` | 0.0378   |

### Production-модель

Версия **4** модели `game_sales_rf`, обучена на **полной выборке** (`X`, `y`),
помечена Stage/тегом `Production`.

**Production run_id:** `473c292857394401a4abfbfdae64d0fe`

Вместе с моделью залогированы:

- сигнатура модели (`infer_signature`);
- пример входных данных (`X.head(5)`);
- `requirements.txt`;
- `selected_features_idx.csv`, `selected_features_names.csv`.

---

## Воспроизводимость

- Все эксперименты проведены с `random_state=42`.
- Единое разбиение данных в ЛР2: `train_test_split(test_size=0.25, random_state=2)`.
- Все прогоны залогированы в MLflow с параметрами и метриками.
- Модель Production зарегистрирована в Model Registry и доступна
  для деплоя в ЛР3.