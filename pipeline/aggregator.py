import numpy as np
import pandas as pd


class HedgeAggregator:
    """
    Freund & Schapire (1997) Hedge algorithm — multiplicative weights update.

    At each time step t:
      1. Receive predictions from N agents: p_1, ..., p_N
      2. Output ensemble prediction: y_hat = sum(w_i * p_i)
      3. Observe actual outcome y_t
      4. Compute squared loss for each agent: L_i = (p_i - y_t)^2
      5. Update weights: w_i <- w_i * exp(-eta * L_i)
      6. Normalize: w_i <- w_i / sum(w_j)

    Regret bound: R_T <= sqrt(T * log(N) / 2)  for eta = sqrt(log(N) / (2*T))
    Interpretation: cumulative loss of ensemble <= best agent in hindsight + O(sqrt(T log N))
    """

    def __init__(self, n_agents: int, eta: float = 0.1):
        self.n_agents = n_agents
        self.eta = eta
        self.weights = np.ones(n_agents) / n_agents   # start uniform
        self.weight_history: list[np.ndarray] = []    # one entry per update step
        self.loss_history: list[np.ndarray] = []

    def aggregate(self, predictions: list[float]) -> float:
        """Weighted average of agent predictions."""
        return float(np.dot(self.weights, predictions))

    def update(self, predictions: list[float], actual: float) -> None:
        """
        Multiplicative weights update.
        Agents that predicted poorly get their weights reduced exponentially.

        Scale-invariance: raw squared-error losses on daily log returns are tiny
        (~1e-4). With those, exp(-eta * loss) ≈ 1 for any sane eta and the weights
        never move — the algorithm becomes a no-op equal-weight average. We therefore
        normalize each step's losses by their mean across agents, so the update
        depends only on *relative* agent performance and eta has a meaningful,
        unit-independent effect. (Standard practice for online learning when the
        loss scale is not naturally in [0, 1].)
        """
        preds = np.array(predictions, dtype=float)
        losses = (preds - actual) ** 2                        # squared error per agent
        scale = losses.mean() + 1e-12                         # per-step loss scale
        norm_losses = losses / scale                          # relative, scale-free
        self.weights *= np.exp(-self.eta * norm_losses)       # multiplicative penalty
        self.weights /= self.weights.sum()                    # renormalize to sum=1
        self.weight_history.append(self.weights.copy())
        self.loss_history.append(losses.copy())

    @staticmethod
    def optimal_eta(T: int, N: int) -> float:
        """
        Theoretically optimal eta = sqrt(log(N) / (2*T)).
        Minimizes the regret bound R_T <= sqrt(T * log(N) / 2).
        Use when T (number of update steps) is known in advance.
        """
        return np.sqrt(np.log(N) / (2 * T))

    def weight_dataframe(self, index=None, agent_names=None) -> pd.DataFrame:
        """Return weight history as a DataFrame for plotting."""
        cols = agent_names or [f"agent_{i}" for i in range(self.n_agents)]
        df = pd.DataFrame(self.weight_history, columns=cols)
        if index is not None:
            df.index = index[:len(df)]
        return df

    def reset(self) -> None:
        """Reset to uniform weights — useful between backtest folds."""
        self.weights = np.ones(self.n_agents) / self.n_agents
        self.weight_history.clear()
        self.loss_history.clear()


class EqualWeightAggregator:
    """
    Naive baseline: simple average across all agent predictions.
    No learning, no adaptation. Used to demonstrate that Hedge > equal weighting.
    """

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.weights = np.ones(n_agents) / n_agents

    def aggregate(self, predictions: list[float]) -> float:
        return float(np.mean(predictions))

    def update(self, predictions: list[float], actual: float) -> None:
        pass   # weights never change

    @property
    def weight_history(self) -> list:
        return []


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    N, T = 4, 10

    # Dummy test: agent 0 is always accurate, agents 1-3 are noisy
    print("── Manual 10-step test ──")
    hedge = HedgeAggregator(n_agents=N, eta=0.1)
    for t in range(T):
        actual = rng.normal(0, 0.01)
        preds = [actual + rng.normal(0, noise)
                 for noise in [0.001, 0.01, 0.02, 0.03]]
        ensemble = hedge.aggregate(preds)
        hedge.update(preds, actual)
        print(f"  t={t+1:2d}  actual={actual:+.4f}  "
              f"ensemble={ensemble:+.4f}  "
              f"weights={np.round(hedge.weights, 3)}")

    print("\nFinal weights:", np.round(hedge.weights, 4))
    print("Agent 0 should have highest weight (most accurate).")
    assert hedge.weights[0] == hedge.weights.max(), \
        "Agent 0 should dominate after 10 steps"
    print("✓ Assertion passed\n")

    # Weight evolution plot
    wdf = hedge.weight_dataframe(
        agent_names=["Accurate", "Noisy-1", "Noisy-2", "Noisy-3"]
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in wdf.columns:
        ax.plot(wdf[col], marker="o", label=col)
    ax.set_title("Hedge Weight Evolution — 10-step Toy Example")
    ax.set_xlabel("Update step")
    ax.set_ylabel("Weight")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/hedge_toy_weights.png", dpi=120)
    print("Plot saved to results/hedge_toy_weights.png")
    plt.show()

    # Optimal eta demo
    T_backtest = 3520   # ~14 years of trading days
    eta_opt = HedgeAggregator.optimal_eta(T=T_backtest, N=4)
    print(f"\nOptimal eta for T={T_backtest}, N=4: {eta_opt:.5f}")
    print(f"Default eta=0.1 vs optimal={eta_opt:.5f} "
          f"({'larger' if 0.1 > eta_opt else 'smaller'} than optimal)")
