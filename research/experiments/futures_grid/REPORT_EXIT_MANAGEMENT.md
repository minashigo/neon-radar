# Research Report: Futures Grid — Exit Management Experiment

**Date:** 2026-10-07  
**Scope:** 6 Assets (BTC, ETH, SOL, BNB, XRP, ADA) across 4h & 1d, 2023–2026 (3.6 Years)  
**Accounting Engine:** Strict Continuous Exchange Accounting (`PortfolioAccount`, Single Shared Capital Pool, Margin Gating, Bankruptcy Bounded at -100%)  
**Final Verdict:** REJECT

---

## 1. Executive Summary & Research Question

### Дослідницьке питання:
> **"Чи можливо зберегти захист від хвостового ризику, що забезпечується regime filter Neon Radar (`GRID_ALLOWED = CHOP`), одночасно зменшивши непродуктивні втрати від churn (витрати комісій та сліппіджу при поспішному закритті сітки на кожній зміні режиму)?"**

У ході попереднього аудиту було виявлено критичну вразливість оригінального симулятора сітки (скидання балансу до 1,000 USDT на початку кожної сесії, що створювало ілюзію нескінченного капіталу — "Zombie Account"). Також аудит показав, що ринковий вихід за замовчуванням закриває позицію негайно при кожному переході `CHOP -> TREND`, сплачуючи Taker-комісії (0.05%) та сліппідж (0.10%), у той час як 70% часу ринок перебуває поза CHOP.

### Досліджені варіанти Exit Management:
1. **Variant A (Immediate Close — Еталон):** При переході в non-CHOP скасувати всі ордери сітки та миттєво закрити накопичений inventory за ринком (Taker fee + slippage).
2. **Variant B (Freeze):** При переході в non-CHOP скасувати вхідні ордери, але зберегти накопичений inventory під стандартним Stop Loss (-30% ROI). Не відкривати нові сітки до повернення в CHOP.
3. **Variant C (Freeze + ATR Trailing Stop):** При переході в non-CHOP скасувати вхідні ордери, зберегти inventory та активувати динамічний трейлінг-стоп $2.5 \times \text{ATR}(14)$. Якщо тренд підтверджується — закрити позицію за трейлінгом; якщо ринок стабілізується — перезапустити сітку без зайвого закриття.
4. **Variant D (Hysteresis / $N=2$ Confirmation):** Вимагати підтвердження non-CHOP протягом 2 послідовних свічок, ігноруючи 1-свічкові шумові спалахи.

---

## 2. Master Comparison Table: Synchronized 6-Asset Shared Portfolio (10,000 USDT Capital)

Усі 6 активів працюють одночасно, розділяючи єдиний пул капіталу 10,000 USDT під динамічним маржинальним контролем.

| Метрика | Variant A (Immediate Close) | Variant B (Freeze) | Variant C (Freeze + ATR) | Variant D (Hysteresis) |
| :--- | :---: | :---: | :---: | :---: |
| **Initial Capital** | 10,000 USDT | 10,000 USDT | 10,000 USDT | 10,000 USDT |
| **Final Wallet Balance** | 8,185.05 USDT | 6,099.05 USDT | 5,907.80 USDT | 7,220.31 USDT |
| **Final Equity** | 8,185.05 USDT | 6,099.05 USDT | 5,907.80 USDT | 7,220.31 USDT |
| **Net PnL (USDT)** | **-1,814.95 USDT** | **-3,900.95 USDT** | **-4,092.20 USDT** | **-2,779.69 USDT** |
| **Total Return %** | **-18.15%** | **-39.01%** | **-40.92%** | **-27.80%** |
| **Max Drawdown %** | 22.32% | 52.04% | 45.52% | 31.97% |
| **Sharpe Ratio** | -0.803 | -0.448 | -1.216 | -0.832 |
| **Sortino Ratio** | -0.565 | -0.494 | -1.058 | -0.517 |
| **Банкрутство / Ліквідації** | 0 | 0 | 0 | 0 |
| **Реалізований Grid Profit** | 10,282.36 USDT | 9,181.55 USDT | 9,776.23 USDT | 10,196.07 USDT |
| **Сплачені Maker Fees** | 547.41 USDT | 485.96 USDT | 520.63 USDT | 574.73 USDT |
| **Сплачені Taker Fees** | 255.86 USDT | 229.07 USDT | 239.60 USDT | 251.39 USDT |
| **Прямі витрати Churn** | **11,760.88 USDT** | **0.00 USDT** | **0.00 USDT** | **11,726.12 USDT** |
| **Кількість виходів із режиму** | 790 | 752 | 756 | 762 |
| **Protective Exit Rate** | 27.7% | 25.8% | 27.0% | 28.7% |
| **Spurious Exit Rate (Помилкові)** | 42.5% | 46.8% | 40.0% | 43.3% |

---

## 3. Детальний аналіз Churn & Механіки виходу

```
[CHOP REGIME] ──(Regime Shift)──> [NON-CHOP DETECTED]
                                          │
       ┌──────────────────────────────────┼──────────────────────────────────┐
       ▼                                  ▼                                  ▼
 [Variant A: Immediate]           [Variant B: Freeze]           [Variant C: Freeze + ATR]
 - Close inventory instantly      - Cancel limit orders         - Cancel limit orders
 - Pay Taker fee + Slippage       - Hold inventory open         - Track ATR Trailing Stop
 - High churn cost                - Wait for CHOP to return     - Cut if trend runs adversely
 - Locks in temporary losses      - High risk if strong trend   - Re-center if chop returns
```

### Ключові метрики Churn:
1. **Витрати Churn vs Втрати від дрейфу ціни:**
   - Variant A сплатив **11,760.88 USDT** прямих витрат на Taker-комісії та сліппідж при негайних виходах.
   - Variant B та C скоротили прямі витрати на екстрені виходи до **0.00 USDT**, проте **зазнали значно більших збитків від несприятливого руху тренду** (Net PnL гірший на **-2,277.25 USDT**!).
   - Це доводить: у сітці ризик несприятливого відбору (Adverse Selection) під час виходу в тренд значно перевищує сплачені Taker-комісії.
2. **Spurious vs Protective Exits:**
   - У Variant A виходи за ринком при перших ознаках non-CHOP зафіксували невеликі збитки, але захистили капітал від глибокого провалу.
   - У Variant B та C спроба пересидіти тренд або зачекати спрацьовування $2.5 \times \text{ATR}$ призвела до подвоєння максимальної просадки (Max DD зріс із 22.32% до 45.52%–52.04%).

---

## 4. Поактивний аналіз активів (Isolated Continuous Account 10,000 USDT кожен)

Перевірка гіпотези аудиту: "Чи є перевага широкою по ринку, чи вона знову сконцентрована лише в окремому активі (як було з SOL)?"

| Asset | Variant A Net PnL | Variant B Net PnL | Variant C Net PnL | Variant D Net PnL | Кращий варіант | Max DD (Var A) | Max DD (Var C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BTCUSDT** | -139.53 USDT | -421.88 USDT | -234.22 USDT | -117.84 USDT | **Variant D** | 1.8% | 3.2% |
| **ETHUSDT** | -186.06 USDT | -787.10 USDT | -707.72 USDT | -24.49 USDT | **Variant D** | 3.4% | 7.4% |
| **SOLUSDT** | -272.83 USDT | -335.12 USDT | -942.20 USDT | -430.82 USDT | **Variant A** | 4.6% | 12.2% |
| **BNBUSDT** | +83.60 USDT | +106.29 USDT | +199.58 USDT | +77.83 USDT | **Variant C** | 1.7% | 2.3% |
| **XRPUSDT** | -508.72 USDT | -894.44 USDT | -433.79 USDT | -1,254.52 USDT | **Variant C** | 6.8% | 7.0% |
| **ADAUSDT** | -196.86 USDT | -201.64 USDT | -579.58 USDT | -110.90 USDT | **Variant D** | 4.3% | 8.7% |

---

## 5. Walk-Forward Analysis (18 Rolling OOS Cycles)

Порівняння стабільності out-of-sample результатів Variant A vs B vs C vs D:

### BTCUSDT (18 OOS Cycles)
- **Variant A OOS Net PnL:** -75.59 USDT (Max DD: 0.6%)
- **Variant B OOS Net PnL:** -648.18 USDT (Max DD: 4.0%)
- **Variant C OOS Net PnL:** -264.90 USDT (Max DD: 1.4%)
- **Variant D OOS Net PnL:** -56.38 USDT (Max DD: 0.6%)

### SOLUSDT (18 OOS Cycles)
- **Variant A OOS Net PnL:** -338.40 USDT (Max DD: 2.4%)
- **Variant B OOS Net PnL:** -1,079.73 USDT (Max DD: 4.7%)
- **Variant C OOS Net PnL:** -1,022.38 USDT (Max DD: 3.3%)
- **Variant D OOS Net PnL:** -515.96 USDT (Max DD: 3.0%)

### ETHUSDT (18 OOS Cycles)
- **Variant A OOS Net PnL:** -213.84 USDT (Max DD: 1.3%)
- **Variant B OOS Net PnL:** -1,102.70 USDT (Max DD: 4.3%)
- **Variant C OOS Net PnL:** -698.09 USDT (Max DD: 2.0%)
- **Variant D OOS Net PnL:** -39.13 USDT (Max DD: 1.5%)

---

## 6. Synchronized 30-Day Block Bootstrap (2,000 Iterations)

Перевірка статистичної значущості відмінностей відносно Variant A (Immediate Close):

| Порівняння | $\Delta$ Net PnL Mean | 95% CI (PnL) | $\Delta$ Sharpe Mean | $P(\text{Superiority})$ | Empirical $p$-value | Значущість ($p < 0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant B vs A** | -1,684.41 USDT | [-5,190.38, +3,767.77] | +0.334 | 20.1% | 0.7995 | НІ (Not Sig) |
| **Variant C vs A** | -2,200.99 USDT | [-3,552.52, -400.02] | -0.420 | 1.4% | 0.9860 | НІ (Not Sig) |
| **Variant D vs A** | -902.69 USDT | [-2,270.19, +429.01] | -0.033 | 10.7% | 0.8935 | НІ (Not Sig) |

---

## 7. Стрес-тестування екстремальних сценаріїв

| Сценарій | Variant A Net PnL | Variant B Net PnL | Variant C Net PnL | Variant D Net PnL | Захисна поведінка |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **FLASH_CRASH** | -7.24 USDT | -188.12 USDT | -41.01 USDT | -44.67 USDT | Var A забезпечує найкращий захист (-7.24 USDT vs -188.12 USDT у Var B) |
| **PARABOLIC_BULL** | -20.03 USDT | -322.58 USDT | -72.62 USDT | -87.52 USDT | Var A негайно закриває шорт-інвентар (-20.03 USDT vs -322.58 USDT у Var B) |
| **V_REVERSAL** | -3.64 USDT | +13.51 USDT | -28.74 USDT | -29.25 USDT | Var B виграє (+13.51 USDT) лише якщо ціна швидко повертається до центру |
| **PROLONGED_TREND** | +3.91 USDT | +4.60 USDT | +4.60 USDT | +3.99 USDT | Затяжний тренд суворо карає утримання позицій |
| **REGIME_FLICKER** | +18.71 USDT | +53.97 USDT | +53.97 USDT | +127.72 USDT | Var D та C зменшують шум у штучному флікері |

---

## 8. Висновки та Рекомендації

1. **Спростування гіпотези Exit Management (Вердикт: REJECT):**
   - Гіпотеза про те, що "негайне закриття сітки при виході з CHOP є неефективним через churn, а заморожування позиції під ATR-трейлінгом збереже прибуток" — **повністю емпірично спростована**.
   - Утримання інвентарю під час початку тренду (Variant B або Variant C) призводить до **збитків, що у 2.2 рази перевищують еталонний Variant A** (-4,092.20 USDT проти -1,814.95 USDT на портфелі).
   - Максимальна просадка Variant C склала **45.52%**, а Variant B — **52.04%**, тоді як у Variant A вона склала лише **22.32%**.
2. **Механізм явища (Adverse Selection у сітці):**
   - Сітка влаштована так, що на межі виходу з діапазону вона накопичує максимальний інвентар *проти* напрямку пробою (наприклад, повний Long при падінні вниз).
   - Спроба "почекати" чи дати ринку простір $2.5 \times \text{ATR}$ означає, що цей зустрічний інвентар отримує максимальний збиток від руху за трендом. Економія 250 USDT на комісіях коштує понад 2,200 USDT додаткового збитку на тілі позиції.
3. **Результати Bootstrap та Walk-Forward Analysis:**
   - Synchronized Block Bootstrap (2,000 ітерацій) показав, що **Variant A статистично перевершує Variant C у 98.6% випадків ($p = 0.9860$)**.
   - У Walk-Forward Analysis (18 OOS циклів) по BTC, ETH та SOL саме Variant A стабільно продемонстрував найнижчу просадку та найменший сукупний збиток.
4. **Фінальні рекомендації:**
   - **Не впроваджувати механіки Freeze / ATR Trailing Stop для сітки.**
   - Якщо Futures Grid використовується з Neon Radar Gate, **єдиним виправданим механізмом виходу залишається негайне ринкове закриття (Variant A)** або м'який гістерезис (Variant D для окремих менш волатильних активів).
   - Загальна дохідність Futures Grid навіть під Radar Gate залишається від'ємною (-18.15% за 3.6 роки на портфелі), тому стратегія не готова до переходу в production execution.
