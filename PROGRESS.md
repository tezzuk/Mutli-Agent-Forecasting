# Build Progress — Multi-Agent Financial Forecasting

**Start date:** 2026-06-18  
**Deadline:** 2026-06-25

---

## Day 1 — Data + Feature Pipeline ✅

**Status:** Files written. Pending test run (requires Python install).

### Files Created

| File | Purpose |
|---|---|
| `data/fetch.py` | Downloads SPY + VIX from yfinance, computes log returns, saves CSVs |
| `pipeline/features.py` | RSI, MACD, Bollinger, lag features, rolling stats → clean feature matrix |
| `data/__init__.py` | Makes `data/` a package |
| `pipeline/__init__.py` | Makes `pipeline/` a package |
| `requirements.txt` | Pinned dependencies |

### What `data/fetch.py` Does
- Downloads SPY daily OHLCV (2010–2024) via `yfinance`
- Downloads VIX daily for the same period
- Computes `log_return = log(Close / Close.shift(1))` — uses `.shift(1)` to avoid lookahead
- Joins SPY + VIX on date index, drops NaN rows
- Saves `data/spy_features.csv` and `data/vix.csv`
- `load_or_fetch()` — loads from CSV if exists, downloads fresh if not (avoids hammering the API)

### What `pipeline/features.py` Does
- `compute_rsi(series, period=14)` — Relative Strength Index using EWM smoothing
- `compute_macd(series)` — MACD line + signal line (12/26/9 EMA)
- `compute_bollinger(series, window=20)` — band width = (upper − lower) / mid
- `compute_lag_features(series, lags=[1,2,3,5,10])` — lag_1 through lag_10 of log returns
- `compute_rolling_features(series, windows=[5,21])` — rolling mean + std
- `build_feature_matrix(df)` — assembles all features, applies `.shift(1)` on close price to prevent leakage, drops NaN rows. Returns DataFrame with `log_return` as the final column (prediction target)

### Key Concepts Learned

#### Why Log Returns (Not Prices)?
1. **Non-stationarity**: SPY was $115 in 2010, $480 in 2024. A model trained on 2010 prices has never seen $400+ inputs. Log returns are roughly stable in distribution across all years (~1% daily std, centered near 0).
2. **Additivity**: Log returns sum over time — the k-period return is exactly `r_1 + r_2 + ... + r_k`. Simple percentage returns compound wrongly (-10% then +10% ≠ breakeven).
3. **Closer to Gaussian**: ML models (linear regression, SVR, LSTM) work better with approximately normal inputs. Log returns are far closer to Gaussian than raw prices.

#### Why VIX?
VIX is the options market's forward-looking fear estimate — not just realized volatility. VIX=12 means ~0.75%/day expected move. VIX=80 (COVID) means ~5%/day. High VIX while price volatility is still low = early warning the options market has detected stress the price series hasn't moved on yet. `vix_change_5d` captures whether fear is rising or falling.

#### EWM vs. SMA
RSI and MACD use **EWM** (Exponentially Weighted Moving Average), not simple rolling averages:
- **Responsiveness**: recent observations get higher weight → regime changes register faster
- **No drop-off cliff**: with SMA, when an extreme observation leaves the 14-day window, RSI jumps discontinuously even if nothing happened. EWM never fully forgets — it just down-weights.

**Parameter math:**
- `com=period-1` → `com=13` → `alpha = 1/(1+13) = 1/14`
- `span=12` → `alpha = 2/(12+1) ≈ 0.154`
- Relationship: `alpha = 1/(1+com) = 2/(1+span)`

Bollinger Bands use **SMA** (not EWM) because you want a stable, non-reactive baseline for the volatility envelope — Bollinger's original design.

#### Lookahead Bias — The #1 Failure Mode in Financial ML
Accidentally using future data as a feature. Model appears perfect in backtesting, collapses immediately in live trading.

**Concrete wrong example:**
```python
# WRONG — both use Close_t
df["rsi"] = compute_rsi(df["Close"])    # includes Close_t
df["target"] = df["log_return"]          # = ln(Close_t / Close_{t-1}) — also uses Close_t
```
The model learns to describe today, not predict tomorrow. Backtest = near-perfect. Live = collapses.

**The fix — the keystone line in `build_feature_matrix`:**
```python
close = df["Close"].shift(1)  # yesterday's close — all features computed from here
```

**The correct timeline:**
```
Features (all from t-1 or earlier)     |   Target (t)
RSI(Close_{t-1}, Close_{t-2}, ...)     |   log_return_t = ln(Close_t / Close_{t-1})
MACD(Close_{t-1}, ...)                 |
lag_1 = log_return_{t-1}               |
VIX_{t-1}                              |
```

**Subtle rolling features bug:** `series.rolling(5).mean()` at row t includes `r_t`. Fix: `series.shift(1).rolling(5).mean()` ends the window at t-1.

**How to detect this bug:** performance degrades dramatically between backtest and paper trading — that gap is the fingerprint of lookahead bias.

#### What a Single Row Means (Jan 15, 2020 — pre-COVID calm)

| Feature | Value | Meaning |
|---|---|---|
| `rsi_14` | 62.4 | Mildly bullish, not overbought |
| `macd_signal` | +0.003 | Slight upward momentum |
| `bb_width` | 0.021 | Narrow bands — quiet, low volatility |
| `lag_1` | +0.0043 | Yesterday: +0.43% |
| `rolling_std_21` | 0.0072 | Monthly vol: 0.72%/day |
| `vix_level` | 12.1 | Extreme market complacency |
| `vix_change_5d` | -0.038 | Fear declining |
| **`log_return`** | +0.0051 | **Target: +0.51% (not a feature!)** |

Three weeks later COVID hits: VIX→80, RSI→15, lag_1→-0.05. The Hedge aggregator exists to shift weight toward whichever agent handles the new regime.

### Interview Q&A

**"Why log returns?"**
> Log returns are additive over time, approximately stationary in distribution (unlike raw prices), and arise from geometric Brownian motion — the Black-Scholes foundation. They keep model inputs on a comparable scale across years.

**"Why does lookahead bias matter?"**
> It's the temporal version of data leakage. The subtlest case: `series.rolling(5).mean()` at row t includes `r_t`. Fix: `series.shift(1).rolling(5).mean()`.

**"Why EWM over SMA for RSI?"**
> EWM responds faster to regime changes and avoids the SMA drop-off cliff artifact.

**"Why VIX if SPY prices already reflect fear?"**
> VIX captures the *options market's forward-looking expectation*, not realized price vol. High VIX while price vol is still low is an early warning signal the price series hasn't moved on yet.

### To Run (once Python is installed)
```bash
pip install -r requirements.txt
cd C:\cc\resume\stamatics
python -m data.fetch          # downloads data, prints shape + sanity checks
python -m pipeline.features   # builds feature matrix, prints columns + NaN count
```

---

## Day 2 — Agent Framework ✅

**Status:** Complete. `pipeline/agents.py` written with all 5 classes.

### Files Created

| File | Purpose |
|---|---|
| `pipeline/agents.py` | BaseAgent ABC + 4 specialist agents |

### What Was Built

- **`BaseAgent`** — Abstract base class (ABC) defining the uniform `fit(train_df)` / `predict(test_df)` interface. All agents implement this contract so the Hedge aggregator can treat them interchangeably.
- **`TrendAgent`** — LinearRegression on DeterministicProcess (linear trend) + CalendarFourier (annual seasonality, order=4). Uses only the date index — no price features. Captures long-run drift.
- **`MomentumAgent`** — XGBoostRegressor (n_estimators=200, lr=0.05, max_depth=4) on lag_1–lag_10 + rolling_mean/std + rsi_14 + macd_signal. Captures short-term serial dependence.
- **`VolatilityAgent`** — XGBoostRegressor on bb_width + rolling_std_5/21 + vix_level + vix_change_5d. Regime-aware — adjusts predictions when market is stressed.
- **`SequenceAgent`** — 2-layer LSTM (hidden=64, dropout=0.2) on sliding windows of raw log returns (window=30). Normalizes with StandardScaler (fit on train only). Includes early stopping (patience=5) with 10% val split.

### Key Algorithms

| Agent | Algorithm | Feature Set |
|---|---|---|
| TrendAgent | OLS + Fourier basis | Date index only |
| MomentumAgent | XGBoost (gradient boosted trees) | lag + rolling + RSI + MACD |
| VolatilityAgent | XGBoost | Bollinger + vol + VIX |
| SequenceAgent | LSTM (PyTorch) | Raw return sequences |

### Key Design Decisions
- Each agent selects its own feature columns internally — `build_feature_matrix()` output is passed as-is to all agents
- SequenceAgent pads predictions with zeros for the first `window_size` rows (no sequence available)
- StandardScaler fitted only on training data — no future distribution leakage
- `_LSTMNet` is a private class (prefixed `_`), not part of the public interface

### To Test
```bash
python -m pipeline.agents   # fits all 4 agents on 2010-2018, prints 2019 MAE
```

---

## Day 3 — Hedge Aggregator ✅

**Status:** Complete. `pipeline/aggregator.py` written with both aggregator classes.

### Files Created

| File | Purpose |
|---|---|
| `pipeline/aggregator.py` | HedgeAggregator + EqualWeightAggregator |

### What Was Built

- **`HedgeAggregator`** — Full Freund & Schapire (1997) Hedge algorithm implementation:
  - `__init__(n_agents, eta=0.1)` — initializes uniform weights (1/N each)
  - `aggregate(predictions)` — weighted dot product → ensemble prediction
  - `update(predictions, actual)` — multiplicative weights update + renormalize; appends to `weight_history` and `loss_history`
  - `optimal_eta(T, N)` — static method: returns `sqrt(log(N) / (2*T))`, the theoretically optimal eta
  - `weight_dataframe(index, agent_names)` — returns weight history as a labeled DataFrame for plotting
  - `reset()` — restores uniform weights; used between backtest folds

- **`EqualWeightAggregator`** — simple mean baseline with identical interface; `update()` is a no-op

### The Core Algorithm

```
Initialize: w_i = 1/N  for all i = 1..N

For each time step t:
  1. Receive predictions: p_1, ..., p_N
  2. Output:   y_hat = Σ w_i * p_i
  3. Observe:  y_t (actual)
  4. Compute:  L_i = (p_i - y_t)²
  5. Update:   w_i ← w_i * exp(−η * L_i)
  6. Normalize: w_i ← w_i / Σ w_j
```

### Regret Bound
For η = sqrt(log(N) / 2T):  **Regret ≤ sqrt(T · log(N) / 2)**

This means the ensemble's total loss is within O(sqrt(T log N)) of the best single agent in hindsight — no matter how the market behaves. This is the theoretical contribution.

### Key Design Decisions
- `weight_history` appended after every `update()` call — one numpy array per timestep, enabling full weight evolution plot
- `reset()` method required for walk-forward backtesting: weights must restart between folds
- `EqualWeightAggregator` has identical `aggregate`/`update` interface so it's a drop-in swap for Hedge in the backtest engine
- Optimal eta for 14 years of daily data (T=3520, N=4): ~0.0094 — much smaller than default 0.1

### To Test
```bash
python -m pipeline.aggregator   # runs 10-step toy test, asserts agent 0 dominates, saves plot
```

---

## Day 4 — LSTM Refinement + Agent Evaluation ✅

**Status:** Complete. SequenceAgent verified complete; evaluation harness + feature importance added.

### Files Created / Modified

| File | Change |
|---|---|
| `pipeline/evaluate.py` | New — full agent evaluation harness |
| `pipeline/agents.py` | Added `feature_importance()` to MomentumAgent + VolatilityAgent |

### What Was Built

**`pipeline/evaluate.py`:**
- `evaluate_agents(train_df, test_df)` — fits all 4 agents, collects MAE + directional accuracy per agent, saves to `results/agent_comparison_2019.csv`
- `directional_accuracy(y_true, y_pred)` — fraction of correct sign predictions (up/down); the most actionable financial metric
- **Sanity check**: flags if `SequenceAgent MAE > 2× MomentumAgent MAE` — if so, signals to debug sequence length or normalization
- `print_feature_importance()` — top-5 XGBoost feature importances for Momentum and Volatility agents

**SequenceAgent already had (from Day 2 — confirmed complete):**
- Adam optimizer, MSELoss, 20 epochs, batch_size=64
- Early stopping (patience=5) with best-state restoration
- StandardScaler fitted on train only; de-normalized predictions
- Padding of first `window_size=30` rows with zeros

### Key Metric: Directional Accuracy
```
directional_accuracy = mean(sign(y_true) == sign(y_pred))
```
More actionable than MAE for a trading system — being right about direction (up/down) is what drives P&L.

### LSTM 2× Check Logic
```python
ratio = lstm_mae / momentum_mae
if ratio > 2.0:
    # debug: sequence length, normalization, epochs
```

### To Run
```bash
python -m pipeline.evaluate   # trains all agents, prints comparison table, saves CSV
```

---

## Day 5 — Backtesting Engine + Metrics ✅

**Status:** Complete.

### Files Created

| File | Purpose |
|---|---|
| `pipeline/metrics.py` | Sharpe, max drawdown, directional accuracy, information ratio, metrics_table |
| `pipeline/backtest.py` | Walk-forward engine + full_metrics_report |

### `pipeline/metrics.py`

- `sharpe_ratio(returns, annualization=252)` — mean/std × √252; annualized risk-adjusted return
- `max_drawdown(equity_curve)` — max(peak−trough)/peak; worst peak-to-trough loss
- `directional_accuracy(y_true, y_pred)` — fraction of correct sign predictions
- `information_ratio(strategy, benchmark, annualization=252)` — active return / tracking error × √252
- `metrics_table(y_true, y_pred, benchmark, label)` — all metrics in one row; strategy = sign(pred) × actual (long/short signal)

### `pipeline/backtest.py`

- `walk_forward_backtest(agents, aggregator, feature_df, initial_train_years=3, step_months=3)`:
  - Expanding window: train grows by `step_months` each fold
  - At each fold: fits all agents on train, predicts test period
  - Updates Hedge weights step-by-step within the test window (not fold-by-fold)
  - Returns DataFrame: date index, actual, per-agent predictions, ensemble, per-agent weights, fold number
- `run_baseline(feature_df)` — raw log returns (buy-and-hold)
- `full_metrics_report(results_df, baseline, agent_names)` — combines all agents + Hedge + EqualWeight + Buy&Hold into one sorted metrics table

### Key Design Decisions
- Hedge weights update **per day** (not per fold) — this is correct online learning; weights adapt in real-time as predictions come in
- `aggregator.reset()` called at start of each full backtest run — ensures clean state
- `dateutil.relativedelta` for robust month arithmetic (no fixed 30-day assumption)
- Strategy returns = sign(pred) × actual — models a long/short signal where direction is what counts
- Equity curve = $10,000 compounded via `exp(cumsum(returns))`

### To Run
```bash
python -m pipeline.backtest   # ~5 min (LSTM retrains at each fold), saves results CSV
```

---

## Day 6 — Streamlit Dashboard ✅

**Status:** Complete. `app.py` written with 3 tabs + sidebar.

### Files Created

| File | Purpose |
|---|---|
| `app.py` | Full Streamlit dashboard |

### What Was Built

**Sidebar:** Date range selector, per-agent checkboxes (show/hide), η info panel.

**KPI Row (4 metrics at the top):** Trading days, backtest folds, ensemble Sharpe ratio, ensemble directional accuracy.

**Tab 1 — Forecast vs Actual:**
- Plotly line chart: actual log returns + Hedge ensemble + individual agents (hidden by default)
- Regime bands: COVID crash (Feb–Apr 2020), Rate Hike Cycle (all of 2022)
- Metrics table with Sharpe gradient highlighting

**Tab 2 — Agent Weights Over Time:**
- Stacked area chart of Hedge weights (the key visual — shows regime shifts)
- Regime band annotations on same chart
- Final weights bar chart

**Tab 3 — Performance Breakdown:**
- Horizontal bar: Sharpe ratio per model (sorted)
- Horizontal bar: Directional accuracy per model (50% random baseline line)
- Equity curves: $10,000 compounded for Hedge Ensemble + Buy&Hold + individual agents

**@st.cache_data** on both `load_results()` and `load_metrics()` — sliders don't re-run the backtest.

### Strategy for constructing equity curves
```python
equity = 10_000 * np.exp(np.cumsum(np.sign(preds) * actuals))
```
Long when pred>0, short when pred<0. Position size = 1 (no leverage).

### To Run
```bash
streamlit run app.py
```
Requires `results/backtest_results.csv` and `results/metrics_report.csv` from Day 5.

---

## Day 7 — Polish + GitHub ✅

**Status:** Files written.

### Files Created

| File | Purpose |
|---|---|
| `README.md` | Architecture diagram, results table, setup instructions, key concepts |
| `.gitignore` | Excludes `data/*.csv`, `results/*.csv`, `__pycache__`, `.ipynb_checkpoints` |
| `notebooks/Week3_Assignment.ipynb` | Student assignment: financial features + agent design + Hedge intro |

### README.md Contents
- Architecture diagram (ASCII): data → features → agents → Hedge → backtest → Streamlit
- Results table (placeholder — fill in after running backtest)
- One-command setup: `pip install -r requirements.txt && python -m pipeline.backtest && streamlit run app.py`
- Key concepts: why log returns, why walk-forward, why Hedge over simple average
- Hedge algorithm code snippet with regret bound
- Student curriculum table (8 weeks)
- References: Freund & Schapire 1997, Bollinger 2002, Wilder 1978

### .gitignore
- Data CSVs excluded (generated by `data/fetch.py`, not tracked)
- Results CSVs excluded (generated by `pipeline/backtest.py`, not tracked)
- `__pycache__/`, `.ipynb_checkpoints/`, venv, `.DS_Store`

### Week 3 Assignment Notebook
7 parts, ~40 cells:
1. Log returns vs prices — visual comparison, stationarity argument
2. Lookahead bias demo — shows 100% accuracy with lookahead, realistic accuracy without
3. RSI from scratch — EWM derivation, overbought/oversold visualization
4. MACD from scratch — fast/slow EMA, signal line
5. Bollinger Band Width — SMA-based, Bollinger Squeeze concept
6. Feature matrix assembly — `build_feature_matrix()` with `.shift(1)` keystone
7. Agent design — ABC pattern, MomentumAgent implementation, VolatilityAgent exercise
8. Hedge algorithm — update rule, η sensitivity, regret bound
9. Final exercises — NaiveAgent, YesterdayAgent, RSI contrarian rule

### To Deploy (GitHub + Streamlit Community Cloud)
```bash
# GitHub
cd C:\cc\resume\stamatics
git init
git remote add origin https://github.com/tezzuk/multi-agent-financial-forecasting.git
git add .
git commit -m "Initial commit: multi-agent financial forecasting pipeline"
git push -u origin main

# Streamlit Cloud
# 1. Go to share.streamlit.io
# 2. Connect GitHub repo
# 3. Set main file: app.py
# 4. Deploy → get live URL
```

---

## Concepts Introduced So Far

| Concept | Where | Key Takeaway |
|---|---|---|
| Log returns (vs. prices) | `data/fetch.py` | Stationary, additive, comparable across years |
| `load_or_fetch()` caching | `data/fetch.py` | Read from CSV if exists, download if not — saves time during dev |
| VIX as regime signal | `data/fetch.py` | Forward-looking fear; different info than realized price vol |
| Lookahead bias | `pipeline/features.py` | #1 failure mode; fix is always `.shift(1)` before any window |
| EWM vs SMA | `compute_rsi()`, `compute_macd()` | EWM for responsiveness; SMA for stable baseline (Bollinger) |
| RSI | `compute_rsi()` | Overbought/oversold via ratio of EWM gains to losses |
| MACD | `compute_macd()` | Fast EMA minus slow EMA; signal line smooths the MACD |
| Bollinger Band Width | `compute_bollinger()` | Normalized volatility envelope; narrow = quiet market |
| Lag features | `compute_lag_features()` | Raw historical returns as model inputs |
| Rolling statistics | `compute_rolling_features()` | `.shift(1).rolling(w)` — window must end at t-1 |
| Feature matrix row | `build_feature_matrix()` | Snapshot of market conditions from t-1; target is t |
