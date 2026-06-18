from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

# ── Feature column sets ──────────────────────────────────────────────────────
MOMENTUM_FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_5", "lag_10",
    "rolling_mean_5", "rolling_mean_21", "rolling_std_5",
    "rsi_14", "macd_signal",
]

VOLATILITY_FEATURES = [
    "bb_width", "rolling_std_5", "rolling_std_21",
    "vix_level", "vix_change_5d",
]


# ── Base class ────────────────────────────────────────────────────────────────
class BaseAgent(ABC):
    """
    Uniform interface for all forecasting agents.
    Each agent receives the full feature DataFrame and selects its own columns.
    """

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Train on rows of the feature matrix (includes 'log_return' column)."""

    @abstractmethod
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Return an array of log-return predictions, one per row of test_df."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ── Agent 1: Trend Agent ──────────────────────────────────────────────────────
class TrendAgent(BaseAgent):
    """
    Captures long-run directional drift + annual seasonality.
    Uses only the date index — no price features.
    Model: LinearRegression on DeterministicProcess trend + CalendarFourier terms.
    """

    def __init__(self, fourier_order: int = 4):
        self.fourier_order = fourier_order
        self._model = LinearRegression()
        self._dp = None

    def _make_dp(self, index: pd.DatetimeIndex) -> DeterministicProcess:
        fourier = CalendarFourier(freq="YE", order=self.fourier_order)
        return DeterministicProcess(
            index=index,
            constant=True,
            order=1,              # linear trend term
            additional_terms=[fourier],
            drop=True,            # drop collinear columns
        )

    def fit(self, train_df: pd.DataFrame) -> None:
        self._dp = self._make_dp(train_df.index)
        X = self._dp.in_sample()
        y = train_df["log_return"].values
        self._model.fit(X, y)

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        # out_of_sample needs the number of steps, but here we pass the index directly
        X = self._dp.out_of_sample(steps=len(test_df), forecast_index=test_df.index)
        return self._model.predict(X)


# ── Agent 2: Momentum Agent ───────────────────────────────────────────────────
class MomentumAgent(BaseAgent):
    """
    Captures short-term serial dependence and momentum patterns.
    Model: XGBoostRegressor on lag features + rolling stats + RSI + MACD.
    """

    def __init__(self):
        self._model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )

    def _select(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in MOMENTUM_FEATURES if c in df.columns]
        return df[cols]

    def fit(self, train_df: pd.DataFrame) -> None:
        X = self._select(train_df)
        y = train_df["log_return"].values
        self._model.fit(X, y)

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return self._model.predict(self._select(test_df))

    def feature_importance(self) -> pd.Series:
        cols = [c for c in MOMENTUM_FEATURES]
        return pd.Series(
            self._model.feature_importances_, index=cols[:len(self._model.feature_importances_)]
        ).sort_values(ascending=False)


# ── Agent 3: Volatility Agent ─────────────────────────────────────────────────
class VolatilityAgent(BaseAgent):
    """
    Regime-aware agent: adjusts predictions based on volatility state.
    Model: XGBoostRegressor on Bollinger width, realized vol, VIX level + change.
    """

    def __init__(self):
        self._model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )

    def _select(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in VOLATILITY_FEATURES if c in df.columns]
        return df[cols]

    def fit(self, train_df: pd.DataFrame) -> None:
        X = self._select(train_df)
        y = train_df["log_return"].values
        self._model.fit(X, y)

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return self._model.predict(self._select(test_df))

    def feature_importance(self) -> pd.Series:
        cols = [c for c in VOLATILITY_FEATURES]
        return pd.Series(
            self._model.feature_importances_, index=cols[:len(self._model.feature_importances_)]
        ).sort_values(ascending=False)


# ── LSTM backbone ─────────────────────────────────────────────────────────────
class _LSTMNet(nn.Module):
    def __init__(self, input_size: int = 1, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ── Agent 4: Sequence Agent ───────────────────────────────────────────────────
class SequenceAgent(BaseAgent):
    """
    Captures non-linear temporal patterns via a 2-layer LSTM.
    Input: sliding windows of raw log returns (window_size=30).
    Normalizes returns before feeding — StandardScaler fit on train only.
    """

    def __init__(self, window_size: int = 30, hidden: int = 64,
                 epochs: int = 20, batch_size: int = 64, lr: float = 1e-3,
                 patience: int = 5):
        self.window_size = window_size
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self._scaler = StandardScaler()
        self._net = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _make_sequences(series: np.ndarray, window: int):
        """Converts a 1-D series into (X, y) sliding windows."""
        X, y = [], []
        for i in range(window, len(series)):
            X.append(series[i - window: i])
            y.append(series[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def fit(self, train_df: pd.DataFrame) -> None:
        returns = train_df["log_return"].values.reshape(-1, 1)
        scaled = self._scaler.fit_transform(returns).flatten()

        X, y = self._make_sequences(scaled, self.window_size)
        # X shape: (N, window_size) → LSTM expects (N, seq_len, input_size)
        X_t = torch.tensor(X).unsqueeze(-1).to(self._device)
        y_t = torch.tensor(y).to(self._device)

        # 10% validation split for early stopping
        val_n = max(1, int(0.1 * len(X_t)))
        X_tr, X_val = X_t[:-val_n], X_t[-val_n:]
        y_tr, y_val = y_t[:-val_n], y_t[-val_n:]

        loader = DataLoader(TensorDataset(X_tr, y_tr),
                            batch_size=self.batch_size, shuffle=False)

        self._net = _LSTMNet(input_size=1, hidden=self.hidden).to(self._device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_val, wait = float("inf"), 0
        best_state = None

        for epoch in range(self.epochs):
            self._net.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimizer.step()

            # early stopping on validation loss
            self._net.eval()
            with torch.no_grad():
                val_loss = criterion(self._net(X_val), y_val).item()

            if val_loss < best_val - 1e-6:
                best_val, wait = val_loss, 0
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state:
            self._net.load_state_dict(best_state)

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        returns = test_df["log_return"].values.reshape(-1, 1)
        scaled = self._scaler.transform(returns).flatten()

        X, _ = self._make_sequences(scaled, self.window_size)
        if len(X) == 0:
            return np.zeros(len(test_df))

        X_t = torch.tensor(X).unsqueeze(-1).to(self._device)
        self._net.eval()
        with torch.no_grad():
            preds_scaled = self._net(X_t).cpu().numpy()

        # de-normalize
        preds = self._scaler.inverse_transform(
            preds_scaled.reshape(-1, 1)
        ).flatten()

        # SequenceAgent can only predict from row window_size onward
        # Pad the first window_size rows with 0 (no prediction available)
        full_preds = np.zeros(len(test_df))
        full_preds[self.window_size:] = preds
        return full_preds


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
    from data.fetch import load_or_fetch
    from pipeline.features import build_feature_matrix
    from sklearn.metrics import mean_absolute_error

    raw = load_or_fetch()
    df = build_feature_matrix(raw)

    train = df[df.index.year <= 2018]
    test  = df[df.index.year == 2019]
    y_test = test["log_return"].values

    agents = [TrendAgent(), MomentumAgent(), VolatilityAgent(), SequenceAgent(epochs=5)]

    for agent in agents:
        print(f"\nTraining {agent.name}...")
        agent.fit(train)
        preds = agent.predict(test)
        mae = mean_absolute_error(y_test, preds)
        print(f"  {agent.name} MAE on 2019: {mae:.6f}")
