"""
simulation.py — Monte Carlo engine and risk metrics.

Provides vectorised GBM simulation and a suite of risk measures
relevant to long-term asset management.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """Container for Monte Carlo output."""
    W_final: np.ndarray        # (n_sim,)   terminal wealth
    trajectories: np.ndarray   # (T+1, n_sim) full paths
    n_sim: int
    T_years: int
    W0: float = 1.0

    def time_axis(self, freq: str = "monthly") -> np.ndarray:
        n_steps = self.trajectories.shape[0] - 1
        return np.arange(n_steps + 1) / (12 if freq == "monthly" else 252)


def simulate(
    w: np.ndarray,
    mu: pd.Series,
    Sigma: pd.DataFrame,
    n_sim: int = 10_000,
    T_years: int = 20,
    W0: float = 1.0,
    seed: int = 42,
) -> SimulationResult:
    """
    Vectorised Monte Carlo simulation of portfolio wealth over T_years.

    Model:  W_{t+1} = W_t * (1 + r_p,t)
    where   r_p,t = w' r_t,  r_t ~ N(mu_m, Sigma_m)

    The simulation uses monthly steps for consistency with the return
    estimation frequency, avoiding compounding approximation errors
    that arise when mixing daily estimation with monthly simulation.

    Cholesky decomposition ensures simulated shocks respect the
    empirical correlation structure across assets.

    Parameters
    ----------
    w       : portfolio weights (n_assets,)
    mu      : annualised expected returns
    Sigma   : annualised covariance matrix
    n_sim   : number of Monte Carlo paths
    T_years : investment horizon in years
    W0      : initial wealth (normalised to 1.0)
    seed    : random seed for reproducibility

    Returns
    -------
    SimulationResult with W_final and full trajectories
    """
    n_steps = T_years * 12
    mu_m = (mu.values / 12).astype(float)
    Sigma_m = (Sigma.values / 12).astype(float)
    L = np.linalg.cholesky(Sigma_m)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_steps, n_sim, len(w)))

    # (n_steps, n_sim, n_assets) → correlated returns
    R = mu_m + (Z @ L.T)
    # (n_steps, n_sim) portfolio returns
    port_ret = R @ w

    # Full trajectories via cumulative product
    traj = np.empty((n_steps + 1, n_sim))
    traj[0] = W0
    traj[1:] = W0 * np.cumprod(1.0 + port_ret, axis=0)

    return SimulationResult(
        W_final=traj[-1],
        trajectories=traj,
        n_sim=n_sim,
        T_years=T_years,
        W0=W0,
    )


def risk_metrics(result: SimulationResult, confidence: float = 0.95) -> dict:
    """
    Compute a full suite of risk and return metrics from a SimulationResult.

    Metrics
    -------
    Median wealth (x W0)
        Preferred over mean because the log-normal distribution of terminal
        wealth is right-skewed; the mean is dominated by tail scenarios.

    CVaR (Conditional Value-at-Risk, also called Expected Shortfall)
        Average return in the worst (1-confidence) fraction of scenarios.
        CVaR is sub-additive and convex — it is a coherent risk measure
        (Artzner et al. 1999) unlike VaR, which ignores tail shape.
        We report it as a return (W_final/W0 - 1) in percentage.

    Probability of ruin P(W_final < W0)
        Probability of ending with less capital than the initial investment
        after T_years. Critical for institutional mandates with capital
        preservation constraints.

    P(x2), P(x3), P(x5)
        Probability of doubling, tripling, quintupling wealth.
        Communicates upside potential in investor-friendly language.

    Annualised CAGR (median)
        Compound Annual Growth Rate computed from median terminal wealth.
        Equivalent to the geometric mean return.
    """
    W = result.W_final
    W0 = result.W0
    T = result.T_years

    total_ret = W / W0 - 1.0
    q_lo = np.percentile(total_ret, (1 - confidence) * 100)
    cvar = float(total_ret[total_ret <= q_lo].mean())

    med = float(np.median(W))
    cagr = float(med ** (1.0 / T) - 1.0)

    return {
        "Median (x W0)":      round(med, 3),
        "CAGR (median)":      f"{cagr * 100:.2f}%",
        "P25 wealth":         round(np.percentile(W, 25), 3),
        "P75 wealth":         round(np.percentile(W, 75), 3),
        f"CVaR {int(confidence*100)}%": f"{cvar * 100:.1f}%",
        "P(ruin)":            f"{(W < W0).mean() * 100:.1f}%",
        "P(x2)":              f"{(W > 2 * W0).mean() * 100:.1f}%",
        "P(x3)":              f"{(W > 3 * W0).mean() * 100:.1f}%",
        "P(x5)":              f"{(W > 5 * W0).mean() * 100:.1f}%",
    }
