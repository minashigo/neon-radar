# Neon Radar

> Професійний десктопний інструмент для аналізу Binance Futures.

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#)

## Можливості (поточного етапу)

- ✅ Завантаження історичних свічок з Binance Futures (публічний REST API, async)
- ✅ Funding rate, Open Interest, 24h ticker з Binance
- ✅ Файловий кеш з TTL (`KlineCache`)
- ✅ Async-Qt bridge: `AsyncWorker` (QThread + asyncio loop)
- ✅ `MarketDataService` — оркестрація запитів через Qt signals
- ✅ Rate limiter (token bucket, weight-based, 1200 req/min)
- ✅ Retries з exponential backoff для 429/5xx/network
- ✅ **7 технічних індикаторів** (SMA, EMA, RSI, MACD, BollingerBands, ATR, VolumeMA)
- ✅ `IndicatorPipeline` — список `IndicatorSpec` → список `IndicatorSeries`
- ✅ Мульти-таймфрейм аналіз (`1D`, `4H` — розширюється)
- ✅ Конфігурація через `config.json` (Pydantic-валідація)
- ✅ Доменні моделі без зовнішніх залежностей (легко тестуються)
- ✅ Ієрархія винятків для зручного оброблення помилок в UI
- ✅ Готова інфраструктура логування (кольоровий консоль + JSON-файл)
- ✅ **Фундамент для аналізу:** Signal, Score, AnalysisResult, MarketState, IndicatorSnapshot
- ✅ **Фундамент для мульти-біржовості:** `ExchangeClient` ABC + стаби для Bybit/OKX/Hyperliquid
- ✅ **Open/Closed для індикаторів:** декоратор `IndicatorRegistry.register`

## Roadmap

| Етап | Стан | Опис |
|------|------|------|
| 1. Фундамент | ✅ | Структура, конфіг, домен, логування |
| 1.5. Scoring foundation | ✅ | Signal/Score/MarketState/Indicator ABC/ExchangeClient ABC |
| 2. Шар даних | ✅ | BinanceClient, mappers, rate limiter, KlineCache, AsyncWorker, MarketDataService |
| 3. Індикатори | ✅ | SMA / EMA / RSI / MACD / BollingerBands / ATR / VolumeMA + IndicatorPipeline |
| 4. UI skeleton | 🔜 | MainWindow, віджети, графіки |
| 5. Інтеграція | 🔜 | Сигнали/слоти, оновлення, кеш |
| 6. Полірування | 🔜 | Тема, шорткати, help |
| 7. Scoring system | 🔜 | Long/Short оцінювання (типи вже готові) |

## Архітектура

Шарова (Clean) архітектура з чіткими межами:

```
        presentation/        ← Qt UI, теми, віджети
                │
                ▼
        application/         ← Оркестрація (services)
                │
                ▼
        domain/              ← Чиста бізнес-логіка, моделі
                ▲
                │
        infrastructure/      ← Binance API, кеш, файли
```

Детальніше — [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Технічні рішення (ADR) — [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Швидкий старт

```bash
# 1. Клонувати
git clone <repo> neon-radar
cd neon-radar

# 2. Створити venv (Python 3.13+)
python3.13 -m venv .venv
source .venv/bin/activate  # або .venv\Scripts\activate на Windows

# 3. Встановити залежності
pip install -e ".[dev]"

# 4. Скопіювати конфіг
cp config.example.json config.json
# Відредагуйте config.json під свої потреби

# 5. Запустити тести
pytest

# 6. Запустити застосунок (поки що тільки entry point; UI в етапі 4)
neon-radar
```

## Конфігурація

`config.json` підтримує такі секції:

- `symbols` — список монет для аналізу (підтримує `enabled` та `note`)
- `timeframes` — таймфрейми для аналізу (`"1d"`, `"4h"`, …)
- `refresh` — інтервал авто-оновлення (секунди)
- `api` — таймаути, ретраї, rate limit
- `cache` — файловий кеш для зменшення навантаження на API
- `ui` — тема, розмір вікна, дефолти
- `logging` — рівень логування, файл, JSON-формат

Усі поля валідуються Pydantic'ом — помилка в конфігу = чітке повідомлення
при старті застосунку, а не падіння в рантаймі.

## Розробка

```bash
# Лінтер
ruff check .
ruff format .

# Типи
mypy src

# Тести з покриттям
pytest --cov=neon_radar --cov-report=term-missing
```

## Ліцензія

MIT
