import numpy as np
import pandas as pd


def sharpe_ratio(returns: np.ndarray, annualization: int = 252) -> float:
    """Annualized Sharpe ratio: mean / std * sqrt(252)."""
    returns = np.asarray(returns, dtype=float)
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(annualization))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum peak-to-trough decline as a positive fraction.
    equity_curve: cumulative value series (e.g. $10,000 compounded).
    """
    curve = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(curve)
    drawdown = (peak - curve) / peak
    return float(drawdown.max())


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with correct sign (up/down)."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def information_ratio(strategy_returns: np.ndarray,
                      benchmark_returns: np.ndarray,
                      annualization: int = 252) -> float:
    """
    Annualized Information Ratio: active return / tracking error.
    IR = mean(strategy - benchmark) / std(strategy - benchmark) * sqrt(252)
    """
    active = np.asarray(strategy_returns) - np.asarray(benchmark_returns)
    if active.std() == 0:
        return 0.0
    return float(active.mean() / active.std() * np.sqrt(annualization))


def metrics_table(y_true: np.ndarray, y_pred: np.ndarray,
                  benchmark: np.ndarray | None = None,
                  label: str = "Model") -> pd.DataFrame:
    """
    Compute all metrics for one agent/ensemble and return as a single-row DataFrame.
    If benchmark is provided, also computes Information Ratio.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Strategy: go long by predicted_return units each day
    # Simple signal: sign(pred) * actual_return (long/short by direction)
    strategy_returns = np.sign(y_pred) * y_true

    equity_curve = 10_000 * np.exp(np.cumsum(strategy_returns))

    row = {
        "Label": label,
        "MAE": round(float(np.mean(np.abs(y_true - y_pred))), 6),
        "Directional Accuracy": round(directional_accuracy(y_true, y_pred), 4),
        "Sharpe Ratio": round(sharpe_ratio(strategy_returns), 3),
        "Max Drawdown": round(max_drawdown(equity_curve), 4),
    }

    if benchmark is not None:
        row["Information Ratio"] = round(
            information_ratio(strategy_returns, benchmark), 3
        )

    return pd.DataFrame([row])
