# Neon Radar — Installation Guide

## Швидкий старт (Windows)

### 1. Вимоги

- **Python 3.13+** — https://www.python.org/downloads/
  - Під час встановлення обов'язково поставте галочку "Add Python to PATH"
- **Git** (опційно) — для клонування репозиторію

### 2. Встановлення

Відкрийте `cmd` або `PowerShell` і виконайте:

```powershell
# Створіть віртуальне середовище
python -m venv .venv
.venv\Scripts\activate

# Встановіть залежності
pip install -r requirements.txt

# Для розробки (тести, lint):
pip install -r requirements-dev.txt
```

### 3. Підготовка конфігурації

```powershell
# Скопіюйте приклади
copy config.example.json config.json
copy config\scoring.example.json scoring_rules.json

# За потреби відредагуйте config.json (список монет, таймфрейми)
# Та scoring_rules.json (ваги правил)
```

### 4. Запуск

**CLI (сканування):**
```powershell
# Список доступних правил
python -m neon_radar.presentation.cli list-rules

# Сканування ринку (потребує інтернет)
python -m neon_radar.presentation.cli scan

# З поясненням per-factor внесків
python -m neon_radar.presentation.cli scan --explain

# Backtest на історичних даних
python -m neon_radar.presentation.cli backtest --start 2024-01-01 --end 2024-12-31
```

**GUI (Radar):**
```powershell
python -m neon_radar.presentation.app
```

або після `pip install -e .`:
```powershell
neon-radar
neon-radar-app
```

### 5. Тестування

```powershell
# Всі тести (319 пройдуть)
pytest

# Конкретний файл
pytest tests/test_scoring_rules.py

# З покриттям
pytest --cov=neon_radar
```

## Troubleshooting (Windows)

### Помилка: "libxkbcommon.so.0 not found"

Ця проблема стосується тільки Linux. На Windows нічого додатково
встановлювати не потрібно — PySide6 поставляє всі DLL.

### Помилка: "ModuleNotFoundError: No module named 'PySide6'"

Ви не активували віртуальне середовище, або pip встановив пакети
в системний Python.

Вирішення:
```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

### Помилка: "No module named 'neon_radar'"

Ви не встановили проєкт. З кореня проєкту:
```powershell
pip install -e .
```

або додайте шлях до PYTHONPATH:
```powershell
$env:PYTHONPATH = "$PWD\src"
```

### GUI не запускається (помилка Qt)

1. Перевірте версію Python: `python --version` — має бути 3.13+
2. Переінсталюйте PySide6: `pip install --force-reinstall PySide6`
3. На Linux без дисплея — `QT_QPA_PLATFORM=offscreen`
4. На macOS — потрібен Python 3.13 від python.org, не від Homebrew

### Binance API: помилки запитів

`BinanceClient` не потребує API-ключа (тільки публічні endpoints).
Якщо отримуєте 429/5xx — перевірте інтернет і ліміти.

Для backtester з великим вікном — може знадобитися VPN/проксі.

## Структура проєкту

```
neon-radar/
├── pyproject.toml            # Build config + dependencies
├── requirements.txt          # Runtime deps (pip install)
├── requirements-dev.txt      # Dev deps (test, lint)
├── README.md                 # Project overview
├── INSTALL.md                # This file
├── config.example.json       # Sample app config
├── config/
│   └── scoring.example.json  # Sample scoring rules
├── src/neon_radar/
│   ├── domain/               # Pure business logic
│   │   ├── models.py         # OHLCV, KlineSeries, Symbol
│   │   ├── indicators/       # 7 indicators (SMA, EMA, RSI, ...)
│   │   ├── scoring/          # FactorRule, Score, Signal, Engine
│   │   └── ...
│   ├── application/          # Orchestration services
│   ├── infrastructure/      # BinanceClient, KlineCache
│   ├── presentation/         # CLI + Qt UI
│   │   ├── cli.py            # neon-radar scan/list-rules/backtest
│   │   ├── main_window.py    # Radar GUI
│   │   └── widgets/          # RankingTable, DetailPanel, ChartWidget
│   └── utils/                # Logging, async bridge
├── tests/                    # 319 tests
├── docs/                     # ARCHITECTURE.md, DECISIONS.md
└── .gitignore
```

## Документація

- `README.md` — огляд проєкту, roadmap, quick start
- `docs/ARCHITECTURE.md` — детальна архітектура, шари, потоки даних
- `docs/DECISIONS.md` — ADR (Architecture Decision Records)
