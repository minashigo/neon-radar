# Архітектура Neon Radar

## 1. Принципи

- **Шарова архітектура (Clean Architecture).** Залежності спрямовані
  всередину: `presentation → application → domain ← infrastructure`.
  Домен не знає ні про Qt, ні про Binance, ні про конфіг-файли.
- **SOLID.** Кожен модуль має одну відповідальність, відкритий до
  розширення, інтерфейси вузькі, абстракції не залежать від деталей.
- **Immutability.** Доменні моделі (`OHLCV`, `KlineSeries`, `AppConfig`)
  — frozen dataclasses. Це виключає цілий клас багів, коли дані
  випадково мутуються між шарами.
- **Async-friendly.** I/O-операції (мережа, диск) виконуються
  асинхронно. UI-потік ніколи не блокується.
- **Тестабельність без UI.** Домен і конфіг тестуються без PySide6.
  Це дає швидкий feedback loop під час розробки.

## 2. Структура каталогів

```
src/neon_radar/
├── __init__.py
├── config/                # Конфігурація
│   ├── models.py          #   Pydantic-моделі
│   └── loader.py          #   Завантажувач config.json
│
├── domain/                # Домен (pure, без I/O)
│   ├── enums.py
│   ├── models.py          #   OHLCV, KlineSeries, Symbol, TickerStats
│   └── exceptions.py
│
├── infrastructure/        # Зовнішні інтеграції
│   └── binance/
│       ├── client.py      #   REST клієнт (httpx async)
│       └── mapper.py      #   DTO → Domain конверсія
│
├── application/           # Оркестрація (use cases)
│   └── services/
│       ├── market_data.py #   Завантаження + кеш
│       └── analysis.py    #   Індикатори + scoring
│
├── presentation/          # PySide6 UI
│   ├── app.py             #   QApplication entry
│   ├── main_window.py
│   ├── widgets/
│   │   ├── symbol_panel.py
│   │   ├── chart_widget.py
│   │   ├── indicator_panel.py
│   │   └── status_bar.py
│   └── theme/
│       └── neon_palette.py
│
└── utils/                 # Утиліти
    ├── logging.py         #   Stdlib logging + кольори/JSON
    └── async_bridge.py    #   QThread ↔ asyncio міст
```

## 3. Потоки даних

Типовий цикл "користувач обрав монету":

```
[User clicks symbol]
        │
        ▼
SymbolPanel.slot
        │
        ▼  Qt signal
MainWindow.on_symbol_selected(symbol, timeframe)
        │
        ▼  Qt signal
MarketDataService.request_klines(symbol, tf)        [QThread, asyncio loop]
        │
        ▼  await
BinanceClient.get_klines(...)                       [httpx async]
        │
        ▼  parse + map
Mapper.to_ohlcv_list(...) → KlineSeries
        │
        ▼  cache hit/miss
CacheStore.get_or_store(...)
        │
        ▼  Qt signal
MainWindow receives KlineSeries
        │
        ▼
AnalysisService.compute_indicators(series)
        │
        ▼  Qt signal
ChartWidget.draw(series + indicators)
        │
        ▼  Qt signal
IndicatorPanel.update(latest_values)
```

**Ключове:** між шарами проходять **тільки доменні об'єкти** (immutable
dataclasses) — ніяких dict, pandas DataFrame, або DTO.

## 4. Async-стратегія

### 4.1 Проблема

PySide6 має власний event loop. `asyncio` — інший. Змішувати їх
неправильно — це джерело "random freezes" і "works on my machine".

### 4.2 Рішення: asyncio в QThread

Кожен сервіс, який працює з I/O (мережа, диск), має **свій** QThread, в
якому живе окремий asyncio event loop.

```
Qt Main Thread (UI)         Service QThread (1)
─────────────────           ──────────────────
QApplication                asyncio.run()
  QMainWindow                 forever:
    SymbolPanel                 while not stop:
      signal ──────────►         await client.fetch(...)
      slot ◄──────────           results = process(...)
        ChartWidget               emit_signal(results)
                                  msleep(interval)
```

**Переваги:**

- Main thread ніколи не блокується на I/O.
- Кожен сервіс ізольований — помилка в одному не валить інші.
- WebSocket можна додати без переробок (просто `await ws.recv()`).
- Тести сервісів — звичайні `asyncio.run()` без Qt.

**Альтернативи, які ми свідомо відхилили:**

| Варіант | Чому відхилили |
|---------|----------------|
| `qasync` | Додаткова залежність, event-loop змішування з Qt |
| `asyncio.run_coroutine_threadsafe` | Працює, але менш контрольовано і складніше тестувати |
| Sync `requests` + QThread | Не масштабується на WebSocket / streaming |

### 4.3 Контракт

Сервіс — це QObject із:

- сигналами для повідомлення UI про результат
- публічними методами-командами (викликаються з main thread)
- внутрішнім asyncio loop, запущеним у `run()` QThread

Приклад використання з main thread:

```python
service = MarketDataService(api_client, cache, config)
service_thread = QThread()
service.moveToThread(service_thread)
service_thread.start()

# Запит
service.request_klines.emit(Symbol("BTCUSDT"), TimeFrame.D1)

# Результат приходить через signal
service.klines_ready.connect(main_window.on_klines)
```

## 5. Обробка помилок

Кожен виняток — підтип `NeonRadarError`. UI ловить саме цей базовий
клас і показує людино-зрозуміле повідомлення + опціонально tech-details
для dev-режиму.

```
NeonRadarError
├── ConfigError         → "Конфіг невалідний. Див. docs."
├── ApiError
│   ├── NetworkError    → "Немає зв'язку з Binance"
│   ├── RateLimitError  → "Перевищено ліміт запитів. Зачекайте."
│   └── ServerError     → "Binance тимчасово недоступний"
├── DataError
│   ├── ParseError      → "Неочікувана відповідь API"
│   └── DataValidationError → "Дані пройшли валідацію, але сумнівні"
└── IndicatorError      → "Недостатньо даних для індикатора"
```

## 6. Тестування

- `tests/test_config.py` — Pydantic-моделі + loader
- `tests/test_domain.py` — domain (без I/O)
- `tests/test_binance_client.py` — API client (mock httpx)
- `tests/test_indicators.py` — індикатори (numpy)
- `tests/test_analysis_service.py` — оркестрація
- `tests/test_ui/` — Qt віджети (pytest-qt)

Покриття ≥80% для `domain/` і `application/` — обов'язкове.
`presentation/` тестується smoke-тестами (вікно відкрилось → ОК).

## 7. Розширюваність

Щоб додати новий індикатор:

1. Додати функцію в `domain/indicators.py` (pure numpy)
2. Зареєструвати її в `IndicatorPipeline`
3. (Опціонально) додати віджет у `presentation/widgets/indicator_panel.py`

Щоб додати нове джерело даних (Bybit, OKX):

1. Створити `infrastructure/bybit/client.py` з тим самим інтерфейсом
2. Створити відповідний mapper
3. Вибрати джерело через конфіг (`api.provider: "binance" | "bybit"`)

Щоб додати новий таймфрейм:

1. Додати значення в `TimeFrame` enum
2. (Опціонально) додати кнопку в UI
