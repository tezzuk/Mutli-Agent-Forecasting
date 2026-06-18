# Multi-Agent Financial Forecasting — Project Bible

## What This Project Is

A mentorship project run under Stamatics, IIT Kanpur. Three mentors (Aayushman Tripathi, Manthan Khetade, Arisht Daiya) teach 8 weeks of content to students, culminating in a full end-to-end pipeline. The project doubles as a resume piece for the mentors.

**The elevator pitch:**
> An online learning system for financial forecasting where specialized agents (trend, momentum, volatility, sequence) compete and adapt in real-time using regret-minimizing game-theoretic aggregation (Hedge algorithm).

**Why it's credible for two domains:**
- **ML**: Implements the Hedge algorithm (Freund-Schapire 1997) with O(sqrt(T log N)) no-regret guarantees, multiple model architectures, walk-forward cross-validation
- **Finance**: Real equity data via yfinance, financial metrics (Sharpe ratio, directional accuracy, max drawdown), backtesting engine, regime-aware model selection

---

## Build Status

**Started:** 2026-06-18 | **Deadline:** 2026-06-25 | **Full log:** `PROGRESS.md`

| Day | Component | Status | Files |
|---|---|---|---|
| 1 | Data + Feature Pipeline | ✅ Written | `data/fetch.py`, `pipeline/features.py` |
| 2 | Agent Framework | ✅ Written | `pipeline/agents.py` |
| 3 | Hedge Aggregator | ✅ Written | `pipeline/aggregator.py` |
| 4 | LSTM Refinement + Evaluation | ✅ Written | `pipeline/evaluate.py` + `agents.py` updated |
| 5 | Backtest + Metrics | ✅ Written | `pipeline/metrics.py`, `pipeline/backtest.py` |
| 6 | Streamlit Dashboard | ✅ Written | `app.py` |
| 7 | Polish + GitHub | ✅ Written | `README.md`, `.gitignore`, `notebooks/Week3_Assignment.ipynb` |

**Pending:** Python install required before any code can be run (`pip install -r requirements.txt`).

---

## Repository Structure (Target)

```
stamatics/
├── CLAUDE.md                  # this file
├── EXECUTION.md               # task checklist
│
├── data/
│   └── fetch.py               # yfinance data ingestion
│
├── pipeline/
│   ├── features.py            # feature engineering (technical indicators + lag)
│   ├── agents.py              # Agent base class + 4 specialist agents
│   ├── aggregator.py          # Hedge algorithm aggregator
│   ├── backtest.py            # walk-forward backtesting engine
│   └── metrics.py             # Sharpe, drawdown, directional accuracy
│
├── app.py                     # Streamlit dashboard
│
├── notebooks/
│   ├── Week1_Assignment.ipynb        # already written — foundations
│   ├── Week2_Assignment.ipynb        # already written — ML for forecasting
│   ├── Week3_Assignment.ipynb        # to write — financial features + agents
│   └── Week4_Final_Project.ipynb    # to write — full pipeline walkthrough
│
└── results/
    └── backtest_results.csv   # saved backtest output for dashboard
```

---

## Architecture

```
Live Data (yfinance)
        │
        ▼
Feature Engineering Layer
  ├── Returns (log returns, not prices)
  ├── Technical: RSI, MACD, Bollinger Band width
  ├── Volatility: rolling std (5d, 21d), GARCH residuals
  └── Calendar: day-of-week, month, quarter
        │
        ▼
Agent Pool (each independently trained, same interface)
  ├── Agent 1 — Trend Agent      : LinearRegression + DeterministicProcess + Fourier
  ├── Agent 2 — Momentum Agent   : XGBoost on lag + rolling features
  ├── Agent 3 — Volatility Agent : XGBoost with GARCH-derived volatility features
  └── Agent 4 — Sequence Agent   : LSTM on raw return sequences
        │
        ▼
Hedge Aggregator (game-theoretic, online learning)
  ├── Maintains weight vector over agents
  ├── Updates multiplicatively: w_i *= exp(-eta * loss_i)
  ├── Normalizes after each step
  └── Guarantees: regret ≤ O(sqrt(T * log(N)))
        │
        ▼
Backtesting Engine
  ├── Walk-forward validation (expanding or rolling window)
  ├── Financial metrics: Sharpe ratio, max drawdown, hit rate
  └── Regime analysis: bull vs. bear agent weight breakdown
        │
        ▼
Streamlit Dashboard
  ├── Agent weight evolution over time (the key visual)
  ├── Forecast vs. actual equity curve
  ├── Metrics table: each agent vs. ensemble vs. buy-and-hold
  └── Regime selector: filter by market period
```

---

## The Core Algorithm: Hedge (Multiplicative Weights Update)

```python
class HedgeAggregator:
    """
    Freund & Schapire (1997) Hedge algorithm.
    Regret bound: R_T <= sqrt(T * log(N) / 2) for eta = sqrt(log(N) / T)
    """
    def __init__(self, n_agents, eta=0.1):
        self.weights = np.ones(n_agents) / n_agents
        self.eta = eta
        self.weight_history = []

    def aggregate(self, predictions):
        return np.dot(self.weights, predictions)

    def update(self, predictions, actual):
        losses = (np.array(predictions) - actual) ** 2
        self.weights *= np.exp(-self.eta * losses)
        self.weights /= self.weights.sum()
        self.weight_history.append(self.weights.copy())
```

This is the beating heart of the project. Every other component serves it.

---

## Dataset

**Primary:** S&P 500 (SPY ETF) daily OHLCV via `yfinance`, 2010–2024
**Secondary:** VIX daily (as a regime indicator, fed as a feature)
**Fallback for students:** Nifty 50 (more relatable for Indian audience)

Use **log returns** as the prediction target, not raw prices. This is standard in finance — prices are non-stationary, returns are closer to stationary.

```python
import yfinance as yf
df = yf.download("SPY", start="2010-01-01", end="2024-12-31")
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
```

---

## Financial Metrics (Primary Evaluation)

| Metric | Formula | Why it matters |
|---|---|---|
| Directional Accuracy | % correct sign prediction | Most actionable — did you call up/down right? |
| Sharpe Ratio | mean(r) / std(r) * sqrt(252) | Risk-adjusted return, standard quant benchmark |
| Max Drawdown | max(peak - trough) / peak | Downside risk, critical for real portfolios |
| Information Ratio | excess return / tracking error | How much skill vs. noise |

**Baseline to beat:** buy-and-hold SPY over the same period.

---

## Agent Specifications

### Agent 1: Trend Agent
- **Model:** LinearRegression with DeterministicProcess (order=1) + CalendarFourier(freq='Y', order=4)
- **Features:** Time index, Fourier terms
- **Captures:** Long-run directional drift + annual seasonality
- **Directly built on:** Week 2 Section 2 (students already coded this)

### Agent 2: Momentum Agent
- **Model:** XGBoostRegressor (n_estimators=200, learning_rate=0.05, max_depth=4)
- **Features:** lag_1 through lag_10, rolling_mean_5, rolling_mean_21, rolling_std_5, RSI_14, MACD_signal
- **Captures:** Short-term serial dependence and momentum patterns
- **Directly built on:** Week 2 Sections 3-4

### Agent 3: Volatility Agent
- **Model:** XGBoostRegressor
- **Features:** Bollinger Band width, realized volatility (rolling_std_21), VIX level, VIX 5d change
- **Captures:** Volatility regime — adjusts predictions when market is stressed
- **New concept for students:** volatility as a predictive signal

### Agent 4: Sequence Agent
- **Model:** LSTM (2 layers, 64 units, dropout=0.2)
- **Features:** Raw return sequences, window=30
- **Captures:** Non-linear temporal patterns too complex for tabular models
- **Directly maps to:** Week 6 of original plan (LSTM/GRU)

---

## Backtesting Protocol

**Walk-forward validation** — no single train/test split:
1. Train on first 3 years (2010–2012)
2. Test on next 3 months
3. Expand training window by 3 months, repeat
4. Collect all test-period predictions into one continuous series
5. Compute metrics on the full out-of-sample record

This is how professional quantitative strategies are validated. A single 80/20 split is not acceptable for financial data.

---

## Student Assignment Mapping

| Week | Topic | Feeds into pipeline |
|---|---|---|
| 1 | Time series foundations, pandas, decomposition | Foundation knowledge |
| 2 | ML for forecasting: linear, XGBoost, hybrid | Agent 1 (Trend), Agent 2 (Momentum) |
| 3 (new) | Financial features: returns, RSI, MACD, Bollinger | Feature engineering layer |
| 4 (new) | Agent framework + Hedge aggregator | agents.py, aggregator.py |
| 5 | Walk-forward backtesting + financial metrics | backtest.py, metrics.py |
| 6 | LSTM agent | Agent 4 (Sequence) |
| 7 | Full pipeline integration + dashboard | app.py |
| 8 | Final submissions, leaderboard, discussion | Results |

---

## Resume Framing

**For ML interviews:**
> "Implemented a multi-agent forecasting system using the Hedge algorithm (online learning, Freund-Schapire 1997) to adaptively weight specialized forecasters. The system achieves no-regret model selection with provable O(sqrt(T log N)) bounds and outperforms the best individual agent in hindsight on S&P 500 return forecasting over a 14-year walk-forward backtest."

**For Finance/Quant interviews:**
> "Built a systematic financial forecasting pipeline with walk-forward backtesting on SPY daily returns (2010–2024). Ensemble of trend, momentum, volatility, and LSTM agents aggregated via multiplicative weights update. Achieved [X] directional accuracy vs. [Y] buy-and-hold Sharpe ratio."

**For teaching/mentorship context:**
> "Designed and delivered an 8-week project course on multi-agent financial forecasting to [N] students at IIT Kanpur (Stamatics), covering time series analysis, ML/deep learning, and online learning theory, culminating in a full pipeline from live data ingestion to Streamlit dashboard."

---

## Key Design Decisions

- **Log returns as target, not prices:** Prices are non-stationary and will cause spurious model fits. Log returns are approximately stationary.
- **Walk-forward, not random split:** Financial data has temporal structure; shuffling introduces future leakage.
- **Hedge over simple average:** Simple averaging ignores differential agent performance. Hedge adapts weights based on realized losses — this is the theoretical contribution.
- **Streamlit over Jupyter for final demo:** Notebooks are for learning; a dashboard is for showing. Recruiters and interviewers can run a Streamlit app without knowing Python.
- **yfinance over toy datasets:** Real data has gaps, corporate actions, regime changes — handling these signals engineering competence.

---

## Constraints

- 7 days to completion from 2026-06-18 (deadline: 2026-06-25)
- Students are at Week 2 of assignments
- No budget for cloud hosting — dashboard runs locally or on Streamlit Community Cloud (free)
- Students have basic Python; LSTM week should be scaffolded heavily
