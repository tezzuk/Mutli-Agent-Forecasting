# Execution Checklist — 7 Days to Completion

**Start date:** 2026-06-18  
**Deadline:** 2026-06-25  
**Owner:** Aayushman Tripathi

Mark tasks with [x] as completed. Work top-to-bottom — later tasks depend on earlier ones.

---

## Day 1 — Data + Feature Pipeline
**Goal:** Real financial data flowing through a clean feature engineering module.

- [ ] Install dependencies: `pip install yfinance pandas numpy scikit-learn xgboost statsmodels streamlit torch`
- [ ] Create `data/fetch.py`
  - [ ] Download SPY daily OHLCV via `yfinance` (2010-01-01 to 2024-12-31)
  - [ ] Download VIX daily for same period (`^VIX`)
  - [ ] Compute `log_return = log(Close / Close.shift(1))`
  - [ ] Drop NaN rows, save to `data/spy_features.csv`
- [ ] Create `pipeline/features.py`
  - [ ] `compute_rsi(series, period=14)` — Relative Strength Index
  - [ ] `compute_macd(series)` — MACD line and signal line (12/26/9)
  - [ ] `compute_bollinger(series, window=20)` — band width = (upper - lower) / mid
  - [ ] `compute_lag_features(series, lags=[1,2,3,5,10])` — shift-based lag columns
  - [ ] `compute_rolling_features(series, windows=[5,21])` — mean + std
  - [ ] `build_feature_matrix(df)` — calls all above, returns clean DataFrame with `log_return` as target column, drops NaN rows
- [ ] Sanity check: print shape, head, and confirm no future leakage (all features use `.shift(1)` minimum)

---

## Day 2 — Agent Framework
**Goal:** 4 working agents with a shared interface, each independently trainable.

- [ ] Create `pipeline/agents.py`
  - [ ] `BaseAgent` class with abstract `fit(X_train, y_train)` and `predict(X_test)` methods
  - [ ] `TrendAgent(BaseAgent)` — wraps LinearRegression + DeterministicProcess + CalendarFourier(freq='Y', order=4)
    - Input: date index only (no price features)
    - Must handle out-of-sample prediction via `dp.out_of_sample(steps=n)`
  - [ ] `MomentumAgent(BaseAgent)` — wraps XGBRegressor
    - Input: lag features + rolling features + RSI + MACD signal
    - Hyperparams: n_estimators=200, learning_rate=0.05, max_depth=4
  - [ ] `VolatilityAgent(BaseAgent)` — wraps XGBRegressor
    - Input: Bollinger width, rolling_std_5, rolling_std_21, VIX level, VIX 5d change
    - Captures regime-dependent prediction adjustments
  - [ ] `SequenceAgent(BaseAgent)` — wraps PyTorch LSTM
    - Input: raw return sequences, window_size=30
    - Architecture: 2-layer LSTM (hidden=64), dropout=0.2, linear output head
    - Include `_make_sequences(series, window)` helper inside class
- [ ] Test each agent: fit on 2010-2018, predict on 2019, print MAE

---

## Day 3 — Hedge Aggregator
**Goal:** Working Hedge algorithm that tracks and adapts agent weights.

- [ ] Create `pipeline/aggregator.py`
  - [ ] `HedgeAggregator` class
    - `__init__(n_agents, eta=0.1)` — initialize uniform weights
    - `aggregate(predictions: list[float]) -> float` — weighted sum
    - `update(predictions: list[float], actual: float)` — multiplicative weight update, normalize
    - `weight_history: list[np.ndarray]` — append current weights after each update
    - `optimal_eta(T, N) -> float` — static method returning sqrt(log(N) / T), the theoretically optimal eta
  - [ ] `EqualWeightAggregator` — simple average baseline to compare Hedge against
- [ ] Manually test on 10 dummy steps: verify weights shift toward lower-loss agents
- [ ] Plot weight evolution on 2019 data (quick matplotlib check — not final dashboard)

---

## Day 4 — LSTM Agent (deeper pass)
**Goal:** LSTM agent actually trained and competitive, not just a placeholder.

- [ ] In `pipeline/agents.py`, finalize `SequenceAgent`:
  - [ ] Training loop: Adam optimizer, MSELoss, 20 epochs, batch_size=64
  - [ ] Early stopping: stop if validation loss doesn't improve for 5 epochs
  - [ ] Normalize returns before feeding to LSTM (StandardScaler, fit on train only)
  - [ ] `predict()` returns de-normalized predictions
- [ ] Train on 2010–2018 data, evaluate on 2019
- [ ] Compare LSTM test MAE vs. MomentumAgent — confirm LSTM is not worse by >2x (if it is, debug sequence length or normalization)

---

## Day 5 — Backtesting Engine + Metrics
**Goal:** Walk-forward backtest across full 2010–2024 period, financial metrics computed.

- [ ] Create `pipeline/metrics.py`
  - [ ] `sharpe_ratio(returns, annualization=252) -> float`
  - [ ] `max_drawdown(equity_curve) -> float`
  - [ ] `directional_accuracy(y_true, y_pred) -> float` — % correct sign
  - [ ] `information_ratio(strategy_returns, benchmark_returns) -> float`
  - [ ] `metrics_table(y_true, y_pred, label="Model") -> pd.DataFrame` — returns all metrics in one row
- [ ] Create `pipeline/backtest.py`
  - [ ] `walk_forward_backtest(agents, aggregator, feature_df, initial_train_years=3, step_months=3) -> pd.DataFrame`
    - Expanding window: train grows, test window is fixed `step_months`
    - At each step: fit all agents on train, predict test period, update Hedge weights with realized losses
    - Collect: date, actual return, each agent's prediction, ensemble prediction, agent weights at each step
    - Return single DataFrame with all of the above
  - [ ] `run_baseline(feature_df) -> pd.Series` — buy-and-hold SPY log returns
- [ ] Run full backtest (this will take a few minutes for LSTM — acceptable)
- [ ] Save results to `results/backtest_results.csv`
- [ ] Print metrics table: each agent vs. ensemble vs. buy-and-hold

---

## Day 6 — Streamlit Dashboard
**Goal:** A shareable, runnable demo that non-coders can interact with.

- [ ] Create `app.py`
  - [ ] Sidebar: date range slider, eta slider for Hedge, agent toggles (include/exclude agents)
  - [ ] Tab 1 — "Forecast vs. Actual"
    - Line chart: actual log returns vs. ensemble prediction vs. individual agents
    - Metrics table below the chart
  - [ ] Tab 2 — "Agent Weights Over Time"
    - Stacked area chart of Hedge weights (the key visual)
    - Annotation: label market regimes (2020 COVID crash, 2022 rate hikes) as vertical lines
  - [ ] Tab 3 — "Performance Breakdown"
    - Bar chart: Sharpe ratio per agent + ensemble + buy-and-hold
    - Bar chart: Directional accuracy per agent + ensemble
    - Equity curve: $10,000 invested following ensemble signal vs. buy-and-hold
  - [ ] Cache data loading with `@st.cache_data` so sliders don't re-run the backtest
- [ ] Test: `streamlit run app.py` — confirm it loads without errors
- [ ] Add regime annotations: 2020-02 to 2020-04 (COVID crash), 2022-01 to 2022-12 (rate hike cycle)

---

## Day 7 — Polish + GitHub
**Goal:** Presentable repo that can be linked on a resume.

- [ ] Initialize git repo in `C:\stamatics`, create `.gitignore` (exclude `data/*.csv`, `__pycache__`, `.ipynb_checkpoints`)
- [ ] Write `requirements.txt` with pinned versions
- [ ] Write `README.md` (keep it tight — 1 architecture diagram, 1 results table, 1 screenshot of dashboard, setup instructions)
- [ ] Push to GitHub under a clean repo name (suggestion: `multi-agent-financial-forecasting`)
- [ ] Deploy dashboard to Streamlit Community Cloud (free, connects to GitHub)
- [ ] Create Week 3 notebook for students: `notebooks/Week3_Assignment.ipynb`
  - [ ] Section 1: Log returns vs. prices (why returns, stationarity)
  - [ ] Section 2: RSI + MACD + Bollinger — code from scratch with exercises
  - [ ] Section 3: Agent class design — have students extend BaseAgent with their own model
  - [ ] Section 4: Introduction to the Hedge algorithm — implement and test on toy data
- [ ] Final check: run `streamlit run app.py` on a clean machine (or ask co-mentor to test)

---

## Stretch Goals (only if Days 1-7 are ahead of schedule)

- [ ] Add sentiment agent: scrape financial news headlines via NewsAPI, use TF-IDF + logistic regression to predict next-day return direction, add as Agent 5
- [ ] Add hyperparameter tuning: Optuna sweep for XGBoost hyperparams within the walk-forward loop (tune on train fold, evaluate on test fold)
- [ ] Add a Kaggle-style leaderboard for students: they submit a CSV of predictions and a script auto-scores directional accuracy
- [ ] GARCH volatility model: fit GARCH(1,1) on log returns, use standardized residuals as input to VolatilityAgent

---

## Key Files Reference

| File | Purpose | Day |
|---|---|---|
| `data/fetch.py` | Download + preprocess SPY + VIX | 1 |
| `pipeline/features.py` | RSI, MACD, Bollinger, lag, rolling | 1 |
| `pipeline/agents.py` | BaseAgent + 4 specialist agents | 2, 4 |
| `pipeline/aggregator.py` | Hedge algorithm + equal-weight baseline | 3 |
| `pipeline/metrics.py` | Sharpe, drawdown, directional accuracy | 5 |
| `pipeline/backtest.py` | Walk-forward engine | 5 |
| `app.py` | Streamlit dashboard | 6 |
| `notebooks/Week3_Assignment.ipynb` | Student assignment | 7 |

---

## Gotchas to Watch For

1. **Future leakage in features:** Every feature must use `.shift(1)` at minimum before it goes into the model. RSI and MACD should be computed on shifted prices, not the current day's close.
2. **LSTM normalization:** Fit the StandardScaler on training data only. Do not fit on the full dataset before splitting — this leaks future distribution info.
3. **Walk-forward with LSTM is slow:** LSTM retrains from scratch at each fold. Either pre-train once and fine-tune at each step, or skip LSTM from walk-forward and use a fixed pre-trained LSTM agent (note this in results).
4. **Hedge eta selection:** eta=0.1 is a safe default. For presentation, also show the theoretically optimal eta = sqrt(log(N)/T) and compare performance.
5. **yfinance API limits:** Don't hammer it in a loop. Download once, save to CSV, load from CSV in all subsequent runs.
6. **Log returns can be negative and small:** Models predicting on this scale need proper scaling. Always check that XGBoost feature importance makes sense before trusting predictions.
