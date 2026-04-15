"""
optimization.py : Mean-variance and behavioral (Prospect Theory) portfolio optimizers.

Two optimization paradigms are implemented side-by-side:
  1. Classical Markowitz (1952): maximise E[r] / sigma   subject to w >= 0, sum(w)=1
  2. Prospect Theory (Kahneman & Tversky, 1979): maximise E[v(W - W0)]
     where v is the asymmetric value function with loss aversion parameter lambda.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Portfolio:
    """Holds weights and derived statistics for a single portfolio."""
    weights: np.ndarray
    tickers: list[str]
    label: str
    expected_return: float = field(init=False)
    volatility: float = field(init=False)
    sharpe: float = field(init=False)
    mu: pd.Series = field(repr=False, default=None)
    Sigma: pd.DataFrame = field(repr=False, default=None)

    def __post_init__(self):
        if self.mu is not None and self.Sigma is not None:
            self.expected_return, self.volatility = portfolio_stats(self.weights, self.mu, self.Sigma)
            rf = float(self.mu.get("BIL", 0.0))
            self.sharpe = (self.expected_return - rf) / self.volatility if self.volatility > 0 else 0.0

    def as_series(self) -> pd.Series:
        return pd.Series(self.weights, index=self.tickers)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def portfolio_stats(
    w: np.ndarray, mu: pd.Series, Sigma: pd.DataFrame
) -> tuple[float, float]:
    """Return (annualised expected return, annualised volatility) for weights w."""
    ret = float(w @ mu.values)
    vol = float(np.sqrt(w @ Sigma.values @ w))
    return ret, vol


def _neg_sharpe(w: np.ndarray, mu: pd.Series, Sigma: pd.DataFrame, rf: float) -> float:
    ret, vol = portfolio_stats(w, mu, Sigma)
    return -(ret - rf) / vol if vol > 1e-10 else 0.0


def _portfolio_vol(w: np.ndarray, mu: pd.Series, Sigma: pd.DataFrame) -> float:
    return portfolio_stats(w, mu, Sigma)[1]


def _run_optimizer(
    objective,
    args: tuple,
    n_assets: int,
    n_restarts: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Wrapper around scipy SLSQP with multiple random restarts.

    Markowitz optimization is non-convex when the long-only constraint is active;
    multiple restarts reduce the probability of converging to a boundary minimum.
    We run `n_restarts` random Dirichlet starting points and keep the best result.
    """
    rng = np.random.default_rng(seed)
    bounds = [(0.0, 1.0)] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    best_val, best_w = np.inf, np.ones(n_assets) / n_assets
    starts = [np.ones(n_assets) / n_assets] + [
        rng.dirichlet(np.ones(n_assets)) for _ in range(n_restarts - 1)
    ]

    for w0 in starts:
        res = minimize(
            objective,
            w0,
            args=args,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-10},
        )
        if res.success and res.fun < best_val:
            best_val, best_w = res.fun, res.x

    # Clip tiny negative weights produced by floating-point error
    best_w = np.clip(best_w, 0, 1)
    return best_w / best_w.sum()


# ---------------------------------------------------------------------------
# Classical Markowitz
# ---------------------------------------------------------------------------

def max_sharpe_portfolio(mu: pd.Series, Sigma: pd.DataFrame, rf: float) -> Portfolio:
    """Tangency portfolio: maximum Sharpe ratio under long-only constraint."""
    n = len(mu)
    w = _run_optimizer(_neg_sharpe, args=(mu, Sigma, rf), n_assets=n)
    return Portfolio(weights=w, tickers=list(mu.index), label="Max Sharpe", mu=mu, Sigma=Sigma)


def min_variance_portfolio(mu: pd.Series, Sigma: pd.DataFrame) -> Portfolio:
    """Global minimum variance portfolio under long-only constraint."""
    n = len(mu)
    w = _run_optimizer(_portfolio_vol, args=(mu, Sigma), n_assets=n)
    return Portfolio(weights=w, tickers=list(mu.index), label="Min Variance", mu=mu, Sigma=Sigma)


def efficient_frontier(
    mu: pd.Series, Sigma: pd.DataFrame, n_points: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trace the efficient frontier by parametric quadratic programming.

    For each target return on the grid [mu_min, mu_max], we solve:
        min  w' Sigma w
        s.t. w' mu = target, sum(w) = 1, w >= 0

    Returns arrays of (volatilities, target_returns) for plotting.
    """
    p_min = min_variance_portfolio(mu, Sigma)
    ret_min, _ = portfolio_stats(p_min.weights, mu, Sigma)

    target_rets = np.linspace(ret_min * 1.001, float(mu.max()) * 0.99, n_points)
    vols, rets = [], []

    for target in target_rets:
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, t=target: portfolio_stats(w, mu, Sigma)[0] - t},
        ]
        res = minimize(
            _portfolio_vol,
            p_min.weights,
            args=(mu, Sigma),
            method="SLSQP",
            bounds=[(0, 1)] * len(mu),
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if res.success:
            vols.append(res.fun)
            rets.append(target)

    return np.array(vols), np.array(rets)


# ---------------------------------------------------------------------------
# Prospect Theory (Kahneman & Tversky, 1979)
# ---------------------------------------------------------------------------

def prospect_value(W: np.ndarray, W0: float = 1.0, lam: float = 2.25, alpha: float = 0.88) -> float:
    """
    Kahneman-Tversky value function.

    The value function v(x) is defined piecewise around a reference point W0:

        v(x) =  x^alpha            if x >= 0   (gains: concave, diminishing sensitivity)
        v(x) = -lambda * |x|^alpha  if x <  0   (losses: convex, loss aversion)

    Empirically estimated parameters (Tversky & Kahneman 1992):
        lambda = 2.25  →  losses hurt ~2.25x more than equivalent gains
        alpha  = 0.88  →  diminishing sensitivity in both domains

    Parameters
    ----------
    W     : array of simulated terminal wealth realisations
    W0    : reference point (initial wealth = 1.0 by convention)
    lam   : loss aversion coefficient (lambda in K&T notation)
    alpha : curvature parameter (same in both domains for simplicity)

    Returns
    -------
    float : mean prospect value across simulations
    """
    x = W - W0
    v = np.where(x >= 0, x**alpha, -lam * ((-x) ** alpha))
    return float(v.mean())


def prospect_theory_portfolio(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    Z_fixed: np.ndarray,
    lam: float = 2.25,
    alpha: float = 0.88,
    W0: float = 1.0,
) -> Portfolio:
    """
    Portfolio maximising expected prospect value over pre-generated shocks Z_fixed.

    Design choice — pre-generated shocks:
        If we re-draw random numbers at each function evaluation, the stochastic
        objective creates a 'rough' landscape (gradient variance >> gradient signal).
        Fixing Z_fixed makes the objective deterministic and smooth, enabling
        gradient-based SLSQP to converge reliably (Common Random Numbers technique,
        see Glasserman 2003, Ch. 7).

    Parameters
    ----------
    Z_fixed : (T, n_sim, n_assets) array of standard-normal shocks, pre-generated once.
    lam, alpha : Prospect Theory parameters.
    """
    n = len(mu)
    mu_m = (mu.values / 12).astype(float)
    L = np.linalg.cholesky((Sigma.values / 12).astype(float))

    def neg_pt(w: np.ndarray) -> float:
        W = _simulate_pt(w, mu_m, L, Z_fixed, W0)
        return -prospect_value(W, W0=W0, lam=lam, alpha=alpha)

    w = _run_optimizer(neg_pt, args=(), n_assets=n)
    return Portfolio(weights=w, tickers=list(mu.index),
                     label=f"Prospect Theory (λ={lam}, α={alpha})", mu=mu, Sigma=Sigma)


def _simulate_pt(
    w: np.ndarray,
    mu_monthly: np.ndarray,
    L: np.ndarray,
    Z: np.ndarray,
    W0: float = 1.0,
) -> np.ndarray:
    """Fast vectorised GBM simulation over pre-generated shocks Z (T × n_sim × n_assets)."""
    W = np.full(Z.shape[1], W0, dtype=float)
    for t in range(Z.shape[0]):
        r = mu_monthly + Z[t] @ L.T   # (n_sim, n_assets)
        W = W * (1.0 + r @ w)         # (n_sim,)
    return W


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def lambda_sensitivity(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    Z_fixed: np.ndarray,
    lambdas: np.ndarray,
    alpha: float = 0.88,
) -> pd.DataFrame:
    """
    Compute optimal portfolios for a grid of loss-aversion parameters lambda.

    This traces the 'behavioral efficient frontier': as lambda increases,
    the investor demands ever more downside protection. The function returns
    a DataFrame of optimal weights (rows = lambda values, cols = tickers).

    Economic question addressed: what is the performance cost of loss aversion?
    i.e., how much median terminal wealth does an investor sacrifice by having
    lambda = 2.25 (empirical) vs lambda = 1 (rational)?
    """
    records = []
    for lam in lambdas:
        p = prospect_theory_portfolio(mu, Sigma, Z_fixed, lam=lam, alpha=alpha)
        records.append(p.weights)
        print(f"  λ={lam:.2f}  →  Sharpe={p.sharpe:.3f}  |  "
              f"Equity share={( p.weights[0] + p.weights[5]) * 100:.1f}%")

    return pd.DataFrame(records, index=lambdas, columns=mu.index)


def myopic_loss_aversion(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    horizons_months: list[int],
    lam: float = 2.25,
    alpha: float = 0.88,
    n_sim: int = 20_000,
    seed_base: int = 42,
) -> pd.DataFrame:
    """
    Replicate the Myopic Loss Aversion experiment of Thaler et al. (1997).

    Key finding: investors who evaluate their portfolio more frequently
    allocate less to risky assets, because short windows contain more
    visible losses - even if long-run expected returns are identical.

    For each horizon h (in months), we:
      1. Generate fresh shocks of length h (different seed per horizon).
      2. Optimise the Prospect Theory objective over that window.
      3. Record the resulting equity share.

    The seed is varied across horizons intentionally: if we fixed the seed,
    a 3-month horizon would see the same first 3 months as the 1-month horizon,
    making the results correlated in a "fallacious" way.
    """
    mu_m = (mu.values / 12).astype(float)
    L = np.linalg.cholesky((Sigma.values / 12).astype(float))
    n = len(mu)
    records = []

    for h in horizons_months:
        rng = np.random.default_rng(seed_base + h)
        Z_h = rng.standard_normal((h, n_sim, n))

        def neg_pt_h(w, Z=Z_h):
            W = _simulate_pt(w, mu_m, L, Z)
            return -prospect_value(W, lam=lam, alpha=alpha)

        w = _run_optimizer(neg_pt_h, args=(), n_assets=n)
        records.append(w)
        eq = (w[mu.index.get_loc("SPY")] + w[mu.index.get_loc("EEM")]) * 100
        print(f"  Horizon {h:3d}m  →  Equity share = {eq:.1f}%")

    return pd.DataFrame(records, index=horizons_months, columns=mu.index)
