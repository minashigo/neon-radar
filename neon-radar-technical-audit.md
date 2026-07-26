# Технічний аудит проєкту Neon Radar

**Репозиторій:** github.com/minashigo/neon-radar
**Дата аудиту:** 26 липня 2026
**Метод:** повний перегляд наданого архіву вихідного коду (`neon-radar-build.zip`) — ~15 633 рядки коду в `src/`, ~7 036 рядків тестів (411 тестових функцій, 55 файлів). Перевірено `domain/`, `application/`, `infrastructure/`, `presentation/`, `tests/`. Живий прогін `pytest`/`mypy` не виконувався (немає мережі для встановлення залежностей; наданий `.venv` — Windows-специфічний і не запускається в Linux-середовищі аудиту); висновки ґрунтуються на статичному читанні коду та AST-аналізі сигнатур/викликів.

---

## 1. Найважливіша знахідка: команда `neon-radar scan` наразі не працює

**Файл:** `src/neon_radar/presentation/cli.py`
**Функції:** `_score_one_symbol` (рядок 348), `_run_scan` (рядок 391)

Сигнатура:
```python
async def _score_one_symbol(
    client: BinanceClient, symbol: Symbol, timeframe, *,
    limit: int, rules: tuple, min_confidence: float,
) -> tuple[Symbol, MarketState, AnalysisResult]:
```

Виклик у `_run_scan`:
```python
_, _, result = await _score_one_symbol(
    client, symbol, _tf_from_str(timeframe),
    limit=limit, rules=rules, min_confidence=scoring_cfg.min_confidence,
    confluence_bonus=scoring_cfg.confluence_bonus,
    confluence_penalty=scoring_cfg.confluence_penalty,
    max_confidence_boost=scoring_cfg.max_confidence_boost,
)
```

`_score_one_symbol` не приймає `confluence_bonus`, `confluence_penalty`, `max_confidence_boost` (немає навіть `**kwargs`). Підтверджено AST-парсером: реальна сигнатура — `args=['client','symbol','timeframe']`, `kwonly=['limit','rules','min_confidence']`. Виклик гарантовано кидає `TypeError` при кожному запуску.

**Чому критично:** це головна команда CLI. Виняток ловиться тут-таки без розрізнення типу:
```python
except Exception as exc:
    logger.warning("Failed to score %s: %s", symbol, exc)
```
Ззовні виглядає як "проблема з мережею для кожного символу", а насправді — неправильний виклик функції. Команда завжди повертає порожню таблицю.

**Порушена документована гарантія:** docstring `application/services/backtester.py` стверджує: *"The rule set is loaded from `ScoringRulesConfig` exactly as `neon-radar scan` does — so what you backtest is what you ship."* `backtester.py` (рядки 312-320) коректно прокидає `confluence_bonus/penalty/max_confidence_boost` в `analyze_series`, `scan` — ні. Задекларований у коді інваріант не виконується.

**Чому не було виявлено:** жоден тест не викликає `_run_scan`/`_score_one_symbol`. `tests/test_cli.py` і `tests/test_cli_explain.py` перевіряють парсер аргументів і форматування виводу, але не сам сценарій `scan` з мокованим `ExchangeClient`.

**Рекомендація:**
1. Додати три параметри в `_score_one_symbol`, прокинути їх у виклик `analyze_series`.
2. Додати `tests/test_cli_scan.py` з мокованим `BinanceClient`, що виконує `_run_scan` end-to-end і перевіряє непорожній результат.
3. Правило процесу: жодна CLI-команда не вважається готовою без хоча б одного інтеграційного тесту, що реально її викликає.

---

## 2. Мертвий код: trade-setup оверлей у графіку ніколи не рендериться

**Файл:** `src/neon_radar/presentation/main_window.py`
**Метод:** `_render_chart_for` (рядки ~290-297)

```python
def _render_chart_for(self, symbol: Symbol) -> None:
    series = self._last_klines.get(symbol)
    if series is None:
        return
    self._last_results.get(symbol)          # результат нікуди не присвоюється
    indicators = self._last_indicators.get(symbol, ())
    trade_setup = None                       # завжди None
    self._chart.render(series, indicators, trade_setup=trade_setup)
```

`self._last_results.get(symbol)` — вираз-оператор, значення відкидається. Судячи з контексту, малося бути щось на кшталт `result = self._last_results.get(symbol); trade_setup = result.trade_setup if result else None`. Замість цього `trade_setup` жорстко захардкоджений як `None`, тож оверлей торгової установки (entry/stop/target) у графіку не малюється незалежно від реальних даних.

**Рекомендація:** дістати `AnalysisResult` з кешу й прокинути реальний `trade_setup`; додати тест, що перевіряє: якщо `_last_results[symbol]` містить непорожній `trade_setup`, `ChartWidget.render` викликається саме з ним.

---

## 3. Системна проблема: мовчазне ковтання винятків в infrastructure-шарі

**Файл:** `src/neon_radar/infrastructure/providers/binance_context.py`

Сім методів (`get_funding`, `get_open_interest`, `get_long_short_ratio`, `get_funding_history`, `get_open_interest_history`, `get_long_short_ratio_history`, `get_taker_flow`) повторюють ідентичний патерн:

```python
try:
    ...
    return context
except Exception:
    return None
```

Без жодного логування. Порівняно з добре продуманою ієрархією винятків у `domain/exceptions.py` (`NetworkError`, `RateLimitError`, `ServerError`, `ParseError`, `DataValidationError`...) — тут вона повністю ігнорується. Мережева помилка, помилка парсингу відповіді Binance чи баг у нормалізаторі виглядають для викликаючого коду однаково: "даних немає". Діагностика збою в проді неможлива.

В одному з методів (`get_taker_flow`, рядок 147) залишився дебаг-принт:
```python
except Exception as e:
    print(f"DEBUG: get_taker_flow failed: {e}")
    return None
```
— слід забутого налагоджувального коду, що суперечить задекларованій в `docs/ARCHITECTURE.md` "готовій інфраструктурі логування".

**Контраст із хорошим прикладом:** `infrastructure/exchanges/binance/client.py::_get_json` робить це правильно — типізує помилки, ретраїть з backoff, поважає `Retry-After` для HTTP 429 (див. розділ 6). Це доводить, що автор **вміє** робити якісний error handling — просто не застосував той самий стандарт послідовно в іншому файлі того ж шару.

**Рекомендація:** заборонити (лінтером, напр. ruff `BLE001`) `except Exception` без `logger.warning(..., exc_info=True)`; прибрати `print()`; розрізняти `NetworkError` (ретрай/повідомлення юзеру) від `ParseError`/`DataValidationError` (баг, потребує алерту).

---

## 4. Архітектура

Реальна структура значно ширша за задекларовану в `docs/ARCHITECTURE.md` (там описано ~15 файлів): повноцінний `domain/trading/` (backtest, paper-trading, walk-forward, regime-classifier, feature-importance, bootstrap), `domain/scoring/` із 14 стратегіями-правилами, `application/services/risk/` (sizing, drawdown, manager), портфельний двигун.

**Дотримання шарів:** напрямок залежностей `presentation → application → domain ← infrastructure` в цілому дотримується — `domain/` не імпортує з `infrastructure/`/`presentation/` у рантаймі (лише в `TYPE_CHECKING`-блоках, що коректно й не виконується під час роботи програми).

**Сильна сторона:** Registry/Strategy pattern для правил скорингу (`domain/scoring/registry.py`) — append-only реєстр із декоратором, перевіркою дублікатів, методом `clear()` для ізоляції тестів. Аналогічно для індикаторів (`IndicatorRegistry`). Справжній Open/Closed: нове правило = новий файл + декоратор, нуль правок у наявному коді.

**Слабкість:** `presentation/cli.py` — **1184 рядки**, один модуль-звалище: парсинг аргументів (`build_parser`, ~250 рядків), 6+ обробників команд, 7 функцій `print_*_report`, JSON-серіалізація, CSV-експорт. `_run_trade_backtest` (рядки 492-616, ~125 рядків) має вкладеність `if args.walk_forward / elif args.feature_analysis / else` з імпортами всередині кожної гілки — висока когнітивна складність, важко тестувати ізольовано (і справді тестується лише периферія).

**Рекомендація:** розбити `cli.py` на `cli/parser.py`, `cli/commands/{scan,backtest,trade_backtest}.py`, `cli/reporting.py`.

---

## 5. Domain-модель — сильна сторона проєкту

**Файл:** `src/neon_radar/domain/models.py`

- `@dataclass(slots=True, frozen=True)` для `OHLCV`, `KlineSeries`, `TickerStats` — immutability + економія пам'яті.
- `OHLCV.__post_init__` валідує інваріанти (`high >= low`, невід'ємні ціни/об'єм, `close_time >= open_time`) — фейл одразу при створенні об'єкта.
- `KlineSeries.__post_init__` перевіряє сортування свічок за часом, кидає `ValueError` з чітким повідомленням (символ + таймфрейм).
- `Symbol` — `str`-підклас із валідацією в `__new__`: типобезпека без накладних витрат серіалізації.

Дрібне зауваження: `KlineSeries.__getitem__` анотований як `-> OHLCV | list[OHLCV]`, але `self.candles` — `tuple`, слайс поверне `tuple`, не `list`. Неточність типів для `mypy`, не баг.

---

## 6. Обробка помилок — зведена таблиця

| Місце | Якість | Коментар |
|---|:-:|---|
| `infrastructure/exchanges/binance/client.py::_get_json` | Висока | Типізовані винятки, retry+backoff, `Retry-After` для 429, чіткий контракт |
| `domain/exceptions.py` | Висока | Продумана ієрархія, задокументована |
| `config/loader.py` | Висока | `pydantic.ValidationError` обгортається в `ConfigError` — fail-fast |
| `infrastructure/providers/binance_context.py` | Низька | 7× `except Exception: return None` без логування, з дебаг-принтом |
| `presentation/main_window.py::_compute_result` | Середня | `except Exception: return None, ()` без логування |
| `presentation/cli.py::_run_scan` | Низька | Ловить `TypeError` як звичайний "збій скорингу" (маскує баг №1) |

Ієрархія винятків побудована якісно, але застосовується непослідовно — приблизно половина кодової бази її ігнорує на користь "ловимо все, повертаємо None/продовжуємо".

---

## 7. Тестування

- 411 тестових функцій, ~7036 рядків тестів проти ~15633 рядків коду (~45% за обсягом — непогане співвідношення для alpha-стадії).
- Домен (`domain/scoring/`, `domain/indicators/`) покритий добре: окремий тест-файл на кожен індикатор і кожне правило скорингу.
- **Прогалина — саме там, де сталися баги №1 і №2:** жодного тесту, що реально виконує `_run_scan`/`_run_backtest`/`_run_trade_backtest` з мокованим `ExchangeClient`; жодного `test_binance_context.py`; для `presentation/` є лише `test_ui_chart.py` і `test_ui_radar.py` — немає `test_main_window.py`.

**Рекомендація:** пріоритет — не нові unit-тести на домен (він і так добре покритий), а mock-based інтеграційні тести на кожну CLI-команду та на `MainWindow._compute_result`/`_render_chart_for`.

---

## 8. Технічний борг і дрібніші code smells

- **Lazy-імпорти всередині функцій** (не `TYPE_CHECKING`): 27 випадків у `cli.py`, 8 у `backtester.py`, 12 у `trade_backtester.py`. Частково виправдано (уникнення циклічних залежностей / швидший старт CLI без важких PySide6-імпортів), але ускладнює читання залежностей модуля.
- **Артефакти кешу в git:** `~/.neon_radar/cache/BTCUSDT_1d.json` — тека `~/` буквально закомічена в корені проєкту, ознака відсутності/помилки в `.gitignore`.
- **Використання `assert` для контролю потоку в продакшн-коді:** `assert result.market_state is not None` (`cli.py`, `main_window.py`), `assert last_error is not None` (`binance/client.py`). З `python -O` асерти вирізаються — при оптимізованому запуску це стане тихим `AttributeError` замість керованого падіння. Краще явний `if ... raise RuntimeError(...)`.

---

## 9. Масштабованість і підтримуваність

- **Дизайн масштабується добре:** `ExchangeClient` ABC + Registry для правил/індикаторів реально дозволяють додавати біржі/індикатори/правила без правок ядра.
- **Продуктивність backtester** (602 рядки, "pre-fetch усе в пам'ять, потім слайсити") прийнятна для дослідницького backtesting, але не масштабується на роки історії з багатьма символами без chunking/streaming — прийнятний trade-off для alpha-стадії.
- **Підтримуваність знижується** через `cli.py` як god-module, непослідовний error handling і розрив між заявленою в docstring поведінкою та реальною (див. п.1) — довіра до документації в коді падає, коли вона вже розходиться з реалізацією в ключовому місці.

---

## 10. Використання патернів проєктування

| Патерн | Де | Якість |
|---|---|:-:|
| Registry / Strategy | `RuleRegistry`, `IndicatorRegistry` | Дуже добре — чисте Open/Closed |
| Value Object | `Symbol`, `Score`, `Signal` (`domain/scoring/value_objects.py`) | Добре |
| Immutable Data Class | `OHLCV`, `KlineSeries`, `TickerStats` | Дуже добре |
| Adapter | `ExchangeClient` ABC + `BinanceClient` | Добре, підтверджено кодом |
| Facade/Orchestrator | `analyze_series` (`application/services/analysis.py`) | Добре — єдина точка входу для CLI/UI/backtest, хоча CLI-шлях (баг №1) цю обіцянку зараз порушує |
| DTO + Mapper | `infrastructure/exchanges/binance/mapper.py`, `providers/binance_dto.py` | Добре, чітке розділення DTO ↔ domain |

---

## Підсумкова оцінка

| № | Критерій | Оцінка /10 | Коментар |
|---|---|:-:|---|
| 1 | Якість архітектури | 8 | Чисті межі шарів, гарний Registry/Strategy; мінус за `cli.py` як god-module |
| 2 | Якість реалізації коду | 6 | Domain — зразковий; presentation/infrastructure — нерівний, реальний баг і мертвий код |
| 3 | Clean Architecture / SOLID | 7 | Напрямок залежностей дотримано; SRP порушено в `cli.py`; OCP реалізовано зразково через реєстри |
| 4 | Обробка помилок | 5 | Відмінна ієрархія винятків, застосована непослідовно |
| 5 | Якість тестування | 6 | Добре покритий домен; критична прогалина на рівні CLI-команд і UI |
| 6 | Масштабованість | 7 | Дизайн дозволяє рости; backtester потребує оптимізації на великих обсягах |
| 7 | Підтримуваність | 6 | Добра документація намірів, але вже розходиться з кодом |
| 8 | Технічний борг | 5 | God-module, закомічені артефакти кешу, дебаг-принт, assert-контроль потоку |
| 9 | Використання патернів | 8 | Registry/Strategy/Adapter/Value Object застосовані свідомо й коректно |
| 10 | Слабкі місця / code smells | — | Див. розділи 1-3, 8 |

### Загальна оцінка: **6.5 / 10**

### Висновок

Це не типовий аматорський crypto-скрипт — доменний шар, реєстри правил/індикаторів і Binance-клієнт написані на рівні, який можна назвати професійним (продумані інваріанти, типізовані винятки, retry/backoff, тестове покриття домену). Але проєкт ще не готовий до продакшну: головна користувацька команда (`scan`) наразі не працює через неправильний виклик функції, є мертвий код у UI (trade setup ніколи не рендериться), а обробка помилок у частині infrastructure-шару систематично ховає збої замість того, щоб їх сигналізувати — і саме ці три речі не покриті тестами.

Коротко: **дизайн — рівня мідл/сеньйора, інтеграція та дисципліна QA — рівня джуніора-мідла.** Перш ніж публікувати проєкт як портфоліо-приклад чи показувати роботодавцю, варто виправити пункти 1-3 (вони швидкі й показові — "знайдено і виправлено бізнес-критичний баг через код-рев'ю"), уніфікувати error handling у `infrastructure/`, додати по одному e2e-тесту на кожну CLI-команду.
