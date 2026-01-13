## Запуск проекта

### 1) Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2) Проверка наличия модели

В проекте должен быть файл:

```
artifacts/sarimax_sber.pkl
```

Если модели ещё нет — обучить и сохранить (скрипт из проекта):

```bash
python scripts/train.py --data-dir data/moex --all --out-dir artifacts
```

### 3) Запуск FastAPI сервиса

```bash
uvicorn main:app --reload
```

После запуска Swagger будет доступен:

* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Что делает сервис

Модель `sarimax_sber.pkl` — это обученный SARIMAX для тикера **SBER**, обученный на целевой переменной **LOG_RETURN** (лог-доходность).
Эндпоинт `/forward` выполняет **inference**: возвращает прогноз `LOG_RETURN` на `horizon` шагов вперёд.

---

## Примеры запросов

### ✅ Успешный запрос (200 OK)

Запрос выполняет прогон модели и возвращает прогноз лог-доходности на 10 шагов.

```bash
curl -X POST "http://127.0.0.1:8000/forward" \
  -H "Content-Type: application/json" \
  -d '{"secid":"SBER","horizon":10}'
```

Пример ответа:

```json
{
  "secid": "SBER",
  "horizon": 10,
  "target": "LOG_RETURN",
  "forecast": [ ... 10 чисел ... ],
  "model": "SARIMAX(1,0,1)"
}
```

---

### ❌ Неверный формат запроса (400 bad request)

По ТЗ: если запрос неверного формата, возвращается 400 `bad request`.

**Пример 1: text/plain**

```bash
curl -X POST "http://127.0.0.1:8000/forward" \
  -H "Content-Type: text/plain" \
  -d "hello"
```

**Пример 2: multipart/form-data**

```bash
curl -X POST "http://127.0.0.1:8000/forward" \
  -F "secid=SBER" \
  -F "horizon=10"
```

Ожидаемый ответ:

```json
{"detail":"bad request"}
```

---

### ❌ Модель не смогла обработать данные (403)

По ТЗ: если модель не смогла выполнить работу, возвращается 403 и сообщение
`"модель не смогла обработать данные"`.

Например, модель обучена только на тикере **SBER**, поэтому другой тикер не поддерживается:

```bash
curl -X POST "http://127.0.0.1:8000/forward" \
  -H "Content-Type: application/json" \
  -d '{"secid":"GAZP","horizon":10}'
```

Ожидаемый ответ:

```json
{"detail":"модель не смогла обработать данные"}
```

---

## История запросов (GET /history)

Эндпоинт возвращает историю всех вызовов `/forward`, сохранённую в базе данных.

```bash
curl "http://127.0.0.1:8000/history"
```

Пример ответа (сокращённо):

```json
[
  {
    "id": 3,
    "created_at": "2026-01-01T20:55:12.123456",
    "secid": "SBER",
    "horizon": 10,
    "payload_len": 40,
    "processing_ms": 12.4,
    "status_code": 200,
    "error": null
  }
]
```

---

## Статистика запросов (GET /stats)

Эндпоинт возвращает:

* статистику времени обработки: `mean`, `p50`, `p95`, `p99`
* характеристики входа: `payload_len` (размер JSON), `horizon`

```bash
curl "http://127.0.0.1:8000/stats"
```

Пример ответа:

```json
{
  "count": 4,
  "timings_ms": {
    "mean": 19.62,
    "p50": 25.55,
    "p95": 29.46,
    "p99": 29.46
  },
  "inputs": {
    "payload_len": {"count": 4, "mean": 39.25, "min": 39, "max": 40},
    "horizon": {"count": 4, "mean": 5.75, "min": 2, "max": 10}
  }
}
```

* `processing_ms` измеряется внутри `/forward` как время выполнения инференса (в миллисекундах)
* `payload_len` используется как характеристика входа (вместо токенов, т.к. вход не текстовый)
* `horizon` — ключевой параметр модели, отражает сложность/объём расчёта

