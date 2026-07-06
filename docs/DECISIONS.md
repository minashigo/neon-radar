# Архітектурні рішення (ADR)

Цей документ фіксує **важливі** технічні рішення, їх контекст і
наслідки. Формат — спрощений ADR (Architecture Decision Record).

---

## ADR-001: Шарова архітектура (Clean Architecture)

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

Потрібен проєкт, який можна розвивати роками без архітектурного боргу.
Типові пастки Python desktop apps — "fat models in Qt classes", бізнес-логіка
в callback'ах, нероздільні UI+API.

### Рішення

Шарова архітектура з чітким напрямком залежностей:

```
presentation → application → domain ← infrastructure
```

Кожен шар має свою відповідальність і **може** залежати тільки від
шарів ближче до центру.

### Наслідки

- ✅ Тести домену без Qt / мережі — швидкі й стабільні
- ✅ Заміна Binance API не зачіпає домен і UI
- ✅ Один responsibility per file — простіше code review
- ⚠️ Більше файлів на старті (але менше на довгій дистанції)
- ⚠️ Потрібно явно мапити DTO → Domain (див. `infrastructure/binance/mapper.py`)

---

## ADR-002: Власний Binance клієнт на `httpx` (async)

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

Є два варіанти: `python-binance` (популярна обгортка) або власний
клієнт на `httpx`.

### Рішення

Власний async клієнт на `httpx`.

### Обґрунтування

- `python-binance` має власні поняття про помилки, rate limits і
  reconnection — їх треба **обходити**, якщо вони не збігаються з
  нашими. Це створює другий шар абстракції, який треба підтримувати.
- WebSocket (для майбутнього real-time) значно простіший у `httpx` +
  `websockets`, ніж через `python-binance`.
- Бібліотеки-обгортки часто відстають від API біржі на кілька місяців.
- Ми контролюємо все: формат логів, retries, backoff, error mapping.

### Наслідки

- ✅ Повний контроль
- ✅ Готовий фундамент для WebSocket
- ⚠️ Більше коду на старті (~300 рядків)
- ⚠️ Треба тримати синхронізацію з API Binance вручну (вони рідко
  ламають public endpoints без deprecation warning)

---

## ADR-003: `pyqtgraph` для графіків, а не `matplotlib`

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

Потрібно малювати candlesticks, обʼєм, індикатори, потенційно real-time.

### Рішення

`pyqtgraph` з власним candle renderer.

### Обґрунтування

| Критерій | pyqtgraph | matplotlib |
|----------|-----------|------------|
| Швидкість | 50–100 fps для 1000 точок | 1–5 fps |
| Qt integration | Native QWidget | Через FigureCanvas — wrapper |
| Real-time | Створений для цього | Працює, але з милицями |
| Стилізація | Програмна, точна | Через style sheets, обмежена |
| Out-of-box candlestick | Немає (пишемо самі) | Є, але важко real-time |

### Наслідки

- ✅ Production-grade performance
- ✅ Real-time оновлення без лагів
- ⚠️ Candles пишемо самі (~100 рядків)
- ⚠️ Друк/export — додаткова робота (для етапу 6)

---

## ADR-004: Власна реалізація індикаторів

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

Альтернативи: `pandas-ta`, `ta`, `finta`, або власні функції.

### Рішення

Власна реалізація в `domain/indicators.py` на чистому numpy.

### Обґрунтування

- EMA, SMA, RSI, MACD, BB, ATR, OBV — це ~200 рядків numpy. Тягнути
  бібліотеку, яка додає 30+ МБ і ще 130 індикаторів, які ми не
  використовуємо — overengineering.
- **Критично:** майбутня система оцінювання потребуватиме **власних**
  комбінованих метрик. Простіше будувати на власних індикаторах, ніж
  обгортати чужі.
- Формули бібліотечних індикаторів можуть трохи відрізнятися
  (різні джерела дають різні "стандарти"). Власна реалізація = одне
  джерело правди.

### Наслідки

- ✅ Повний контроль над формулами
- ✅ Немає ризику "оновили бібліотеку — все зламалось"
- ✅ Можна оптимізувати під наші патерни використання
- ⚠️ Формули треба підтримувати (але це ~10 рядків на індикатор)
- ⚠️ Потрібно тестувати проти відомих значень (див. `tests/test_indicators.py`)

---

## ADR-005: asyncio в QThread, без `qasync`

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

PySide6 має власний event loop. `asyncio` — інший. Треба вирішити, як
вони співіснують.

### Рішення

Кожен сервіс з I/O має **виділений QThread** з власним asyncio event
loop. UI thread — тільки UI.

### Відхилені альтернативи

| Варіант | Проблема |
|---------|----------|
| `qasync` | Змішування Qt і asyncio event loops — складно дебажити, окрема залежність |
| `asyncio.run_coroutine_threadsafe` | Працює, але менш контрольовано, важко тестувати |
| Sync `requests` + QThread | Блокує event loop при багатьох запитах; не масштабується на WebSocket |
| `qthreadworker` patterns | Додаткова залежність; наша мета простіша |

### Наслідки

- ✅ Чистий separation of concerns
- ✅ Main thread ніколи не блокується
- ✅ WebSocket додається тривіально (`await ws.recv()` у тому ж loop)
- ✅ Тести сервісів — `asyncio.run()` без Qt
- ⚠️ Треба явно сигналити в main thread через Qt signals
- ⚠️ Трохи більше boilerplate на старті

---

## ADR-006: Pydantic v2 для конфігурації

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

Конфігурація з `config.json` має бути валідованою, типобезпечною і
зрозумілою при помилках.

### Рішення

Pydantic v2 моделі з `frozen=True` і `extra="forbid"`.

### Обґрунтування

- Валідація **на старті** застосунку, а не в рантаймі (fail fast).
- IDE autocomplete на полях конфігу.
- Чіткі повідомлення про помилки (Pydantic дає `loc` + `msg`).
- Frozen config = неможливо випадково мутувати.

### Наслідки

- ✅ Type-safe конфіг
- ✅ Зрозумілі повідомлення про помилки
- ⚠️ Pydantic — додаткова залежність (~2 МБ)
- ⚠️ Імпорти Pydantic не повинні потрапити в domain (тримаємо в `config/`)

---

## ADR-007: stdlib logging, не loguru/structlog

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

Логування має бути простим, без зайвих залежностей, і при цьому давати
гарний console-вивід і структурований JSON для продакшну.

### Рішення

`logging` з stdlib + власний `_ColorFormatter` і `_JsonFormatter`.

### Обґрунтування

- Stdlib = 0 залежностей.
- Qt вже використовує stdlib logging — інтегрується без конфліктів.
- Для desktop app loguru додає API, який ми не використовуємо на 90%.

### Наслідки

- ✅ Жодних додаткових залежностей
- ✅ Кольоровий консоль + JSON-файл
- ⚠️ Якщо колись треба буде трейсинг/APM — можемо додати OpenTelemetry

---

## ADR-008: `src/` layout (не flat)

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

Python packaging має два основні layout'и: flat (`module/`) і src (`src/module/`).

### Рішення

`src/neon_radar/` layout.

### Обґрунтування

- Захищає від випадкового імпорту з cwd (`import neon_radar` працює
  тільки якщо встановлено в env).
- Стандарт для serious Python проєктів (PyPA рекомендує).
- Краще для CI і packaging.

### Наслідки

- ✅ Правильна ізоляція
- ✅ `pip install -e .` працює як очікується
- ⚠️ Імпорти в тестах трохи довші (але це добре — явно показує шар)

---

## ADR-009: Python 3.13+ як мінімальна версія

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

Користувач явно вимагає 3.13+.

### Рішення

`requires-python = ">=3.13"` у `pyproject.toml`. Використовуємо сучасні
фічі 3.13:

- `match` statements
- `type` aliases (PEP 695) — обережно, для складних cases
- покращені error messages
- `*`-unpacking
- `from __future__ import annotations` всюди для forward refs

### Наслідки

- ✅ Нові фічі мови доступні
- ✅ Кращі error messages для dev experience
- ⚠️ Користувачі на 3.11/3.12 не зможуть запустити (прийнятно для нового проєкту)

---

## ADR-010: Exchange abstraction (`ExchangeClient` ABC)

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

Вимога: додати в майбутньому Bybit, OKX, Hyperliquid тощо без
переписування бізнес-логіки. Без абстракції кожна біржа просочується
в кожен шар (config, services, presentation).

### Рішення

Один інтерфейс — :class:`ExchangeClient` ABC — у
`infrastructure/exchanges/base.py`. Конкретні реалізації:

* `infrastructure/exchanges/binance/`
* `infrastructure/exchanges/bybit/` (стаб)
* `infrastructure/exchanges/okx/` (стаб)
* `infrastructure/exchanges/hyperliquid/` (стаб)

Application-сервіси залежать тільки на ABC, ніколи на конкретний клас.

### Наслідки

- ✅ Додати нову біржу = новий файл у `exchanges/<name>/` + implements ABC
- ✅ Тести application рівня — мок-клієнт, без реальної мережі
- ✅ Презентація ніколи не бачить exchange-специфічних типів
- ⚠️ Інтерфейс «вузький» — додавання методу = оновлення всіх реалізацій.
  Це свідома ціна: запобігає дрифту API.

---

## ADR-011: Indicator registry (Open/Closed)

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

Вимога: нові індикатори без зміни існуючої логіки. Якщо кожен
індикатор вимагає редагування "switch case" або списку імпортів —
це класичне порушення Open/Closed.

### Рішення

:class:`Indicator` ABC + декоратор :meth:`IndicatorRegistry.register`.

```python
@IndicatorRegistry.register("rsi")
class RSI(Indicator):
    KIND = IndicatorKind.OSCILLATOR
    def __init__(self, period: int = 14): ...
    def compute(self, series: KlineSeries) -> IndicatorSeries: ...
```

Щоб додати новий індикатор — створюємо **один** новий файл. Все
інше (engine, UI, persistence) підхоплює автоматично.

### Наслідки

- ✅ Істинний Open/Closed: extension = новий файл, zero modifications
- ✅ Тести індикаторів ізольовані (кожен у своєму файлі)
- ✅ UI може запитати реєстр: "які індикатори доступні?" без знання про конкретні класи
- ⚠️ Глобальний registry = глобальний стан. Для тестів є `clear()`.

---

## ADR-012: Probability-based scoring (Score / Signal / Confidence)

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

Вимога: оцінка ймовірності, не "EMA перетнула EMA". Кожен фактор
повинен мати вагу, кінцевий результат — загальний бал + рівень
впевненості.

### Рішення

* :class:`Score` — фінальний бал `[-1, +1]` + confidence `[0, 1]`
  + компоненти long/short для пояснення.
* :class:`Signal` — внесок одного фактора: `value [-1, +1]`,
  `weight [0, 1]`, `confidence [0, 1]`, `evidence` для UI.
* :class:`FactorRule` ABC — окремий клас правила з `evaluate(state)`.
  Конкретні правила реалізуються в Етапі 7, але інтерфейс
  зафіксований зараз.
* Engine (Етап 7) обчислює `Score` як зважену суму сигналів.

### Наслідки

- ✅ Кожне рішення має пояснення (`evidence`)
- ✅ Можна вмикати/вимикати правила без зміни engine
- ✅ Confidence агрегується з урахуванням кількості сигналів та їх
  згоди між собою
- ⚠️ Ваги — це конфіг. У Етапі 7 додамо `scoring_rules.json`
- ⚠️ Engine ще не реалізований; типи готові, логіка — в Етапі 7

---

## ADR-013: Multi-factor MarketState

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

Скоринг потребує: candles (поточного TF) + вищий TF + індикатори +
24h ticker + funding + open interest. Все це в одному об'єкті.

### Рішення

:class:`MarketState` — композиція:

```python
MarketState(
    symbol: Symbol,
    timestamp: int,
    primary_series: KlineSeries,         # обов'язково
    higher_tf_series: KlineSeries | None,
    indicator_series: tuple[IndicatorSeries, ...],
    ticker: TickerStats | None,
    funding_rate: FundingRate | None,
    open_interest: OpenInterest | None,
)
```

### Наслідки

- ✅ Один об'єкт передається через шари, немає "10-аргументних" функцій
- ✅ Контракт engine: "дай мені MarketState, я поверну AnalysisResult"
- ✅ Опціональні поля — для випадків, коли funding/OI недоступні
  (spot-only біржі)
- ⚠️ Validation: `higher_tf_series.timeframe` має бути **строго вищим**
  за primary. Перевіряється в `__post_init__`.

---

## ADR-014: Async-Qt bridge через QThread + asyncio

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

`ExchangeClient` async (httpx). Qt main thread неблокуючий. Треба
з'єднати два event loops.

### Рішення

`:class:AsyncWorker` — QThread, в якому `run()` запускає окремий
`asyncio` loop. Сервіси (:class:`MarketDataService`) тримають
AsyncWorker і через нього відправляють корутини в worker-thread
event loop. Результати повертаються через Qt signals (Qt
автоматично використовує QueuedConnection для cross-thread signals).

### Відхилені альтернативи

| Варіант | Проблема |
|---------|----------|
| `qasync` | Додаткова залежність |
| `asyncio.run_coroutine_threadsafe` без QThread | Складніше тестувати, немає нативної Qt-інтеграції |
| Sync `requests` + QThread | Блокує event loop при багатьох запитах; не масштабується на WebSocket |

### Наслідки

- ✅ Async-friendly (готовність до WebSocket без переробок)
- ✅ Main thread ніколи не блокується на I/O
- ✅ Тести — мок ExchangeClient, без реальної мережі
- ⚠️ Lifecycle management: потрібно start() і stop() на сервісах

---

## ADR-015: Binance REST через httpx (без `python-binance`)

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

Потрібен конкретний клієнт Binance Futures public API.

### Рішення

Власний `BinanceClient(ExchangeClient)` на `httpx.AsyncClient` з
централізованим методом `_get_json()` який інкапсулює:

* retries (exponential backoff для 429/5xx/network)
* rate limiting (TokenBucketRateLimiter)
* HTTP → domain exception mapping

### Наслідки

- ✅ Повний контроль над HTTP-логікою
- ✅ Mapper — pure-функції, тестуються окремо
- ✅ Тести з `httpx.MockTransport` — без мережі
- ⚠️ Retry-логіка потребує ретельного тестування (є тести)
- ⚠️ При зміні API Binance треба оновлювати mapper + client

---

## ADR-016: Filesystem JSON cache (а не SQLite / Redis)

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

Кеш потрібен для зменшення навантаження на Binance API.

### Рішення

`KlineCache` — по одному JSON файлу на `(symbol, timeframe)`. TTL
через `st_mtime`. Атомарний write через `temp + rename`.

### Відхилені альтернативи

| Варіант | Проблема |
|---------|----------|
| SQLite | Додаткова залежність; надмірно для одного типу даних |
| Redis | Зовнішній процес; не підходить для desktop app |
| in-memory dict | Втрачається при перезапуску; немає TTL |
| Parquet | Бінарний формат, не дебажиться з `cat` |

### Наслідки

- ✅ Файли — `cat` / `jq`-debuggable
- ✅ Атомарний write (temp + rename)
- ✅ Fail-soft: corrupt file → warning + повернення None
- ⚠️ Багато дрібних файлів для багатьох монет × таймфреймів
  (10 монет × 2 TF = 20 файлів — прийнятно)

---

## ADR-017: Custom numpy indicators (без `pandas-ta`)

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

Потрібні базові технічні індикатори для аналізу.

### Рішення

Сім вбудованих індикаторів реалізовано вручну на чистому numpy:

* `SMA` (Simple Moving Average)
* `EMA` (Exponential Moving Average)
* `RSI` (Wilder's Relative Strength Index)
* `MACD` (3 виходи: macd/signal/histogram)
* `BollingerBands` (3 виходи: upper/middle/lower)
* `ATR` (Wilder's smoothing of True Range)
* `VolumeMA` (Moving Average обʼєму)

Кожен — окремий файл у `domain/indicators/`. Загальні numpy-утиліти —
у приватному `_numpy_helpers.py` (SMA, EMA, Wilder, rolling std).

### Обґрунтування

* Повний контроль над формулами (критично для майбутнього scoring)
* Zero external dependencies для обчислень
* Формули добре відомі, ~10 рядків кожна
* Індикатор «з нуля» = один файл, zero modifications в існуючому коді
* Тести проти hand-computed значень — без залежностей

### Наслідки

- ✅ Жодних залежностей на бібліотеки індикаторів
- ✅ Прозорі формули, можна аудитувати
- ✅ Швидкі: numpy O(n), на 500 свічках < 1 мс
- ⚠️ Якщо потрібен екзотичний індикатор (Ichimoku, Heikin-Ashi) — реалізуємо
  самі. Це може бути 50-100 рядків на індикатор.
- ⚠️ Індикатори розраховуються на запит (batch). Real-time streaming —
  окрема задача на майбутнє.

**Дата:** 2026-06-26
**Статус:** Accepted

### Контекст

Скоринг потребує: candles (поточного TF) + вищий TF + індикатори +
24h ticker + funding + open interest. Все це в одному об'єкті.

### Рішення

:class:`MarketState` — композиція:

```python
MarketState(
    symbol: Symbol,
    timestamp: int,
    primary_series: KlineSeries,         # обов'язково
    higher_tf_series: KlineSeries | None,
    indicator_series: tuple[IndicatorSeries, ...],
    ticker: TickerStats | None,
    funding_rate: FundingRate | None,
    open_interest: OpenInterest | None,
)
```

### Наслідки

- ✅ Один об'єкт передається через шари, немає "10-аргументних" функцій
- ✅ Контракт engine: "дай мені MarketState, я поверну AnalysisResult"
- ✅ Опціональні поля — для випадків, коли funding/OI недоступні
  (spot-only біржі)
- ⚠️ Validation: `higher_tf_series.timeframe` має бути **строго вищим**
  за primary. Перевіряється в `__post_init__`.
