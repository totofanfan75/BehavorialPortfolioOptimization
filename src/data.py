"""
data.py — Asset universe loading and preprocessing.

Downloads monthly log-returns for a diversified 7-asset universe
via yfinance. Falls back to reproducible synthetic data if offline.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TICKERS: list[str] = ["SPY", "AGG", "GLD", "VNQ", "TLT", "EEM", "BIL"]

ASSET_LABELS: dict[str, str] = {
    "SPY": "US Equities",
    "AGG": "US Bonds (Agg.)",
    "GLD": "Gold",
    "VNQ": "US Real Estate",
    "TLT": "LT US Treasuries",
    "EEM": "Emerging Markets",
    "BIL": "Cash / T-Bills",
}


def load_returns(
    tickers: list[str] = TICKERS,
    start: str = "2010-01-01",
    end: str = "2026-01-01",
    freq: str = "ME",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load monthly log-returns, annualised mean vector mu and covariance Sigma.

    Monthly frequency is preferred over daily for two reasons:
      1. Removes microstructure noise and bid-ask bounce effects.
      2. Avoids the Epps (1979) effect where daily cross-correlations
         are downward biased due to non-synchronous trading hours.

    Parameters
    ----------
    tickers : list of ticker symbols
    start, end : date range (ISO format)
    freq : pandas offset alias for resampling (default 'ME' = month-end)
    seed : random seed for synthetic fallback

    Returns
    -------
    returns : pd.DataFrame   — monthly log-returns
    mu      : pd.Series      — annualised expected returns
    Sigma   : pd.DataFrame   — annualised covariance matrix
    """
    try:
        import yfinance as yf

        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)[
            "Close"
        ]
        prices = raw.ffill().dropna().resample(freq).last()
        print(f"[data] Downloaded {len(prices)} monthly observations from Yahoo Finance.")
    except Exception:
        print("[data] yfinance unavailable — using synthetic data (seed={seed}).")
        prices = _synthetic_prices(tickers, seed=seed)

    returns = np.log(prices / prices.shift(1)).dropna()
    mu = returns.mean() * 12          # annualise
    Sigma = returns.cov() * 12        # annualise
    return returns, mu, Sigma, prices


def _synthetic_prices(tickers: list[str], n_months: int = 192, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic monthly prices via a correlated GBM."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")

    # Annualised calibration (rough proxies for 2010-2025)
    mu_ann = np.array([0.10, 0.03, 0.05, 0.08, 0.04, 0.09, 0.015])
    vol_ann = np.array([0.15, 0.04, 0.16, 0.18, 0.12, 0.20, 0.005])
    corr = np.array(
        [
            [1.00, -0.10, 0.05, 0.70, 0.00, 0.80, -0.05],
            [-0.10, 1.00, 0.10, -0.05, 0.85, -0.05, 0.10],
            [0.05, 0.10, 1.00, 0.05, 0.10, 0.10, -0.05],
            [0.70, -0.05, 0.05, 1.00, 0.00, 0.60, -0.05],
            [0.00, 0.85, 0.10, 0.00, 1.00, 0.00, 0.05],
            [0.80, -0.05, 0.10, 0.60, 0.00, 1.00, -0.05],
            [-0.05, 0.10, -0.05, -0.05, 0.05, -0.05, 1.00],
        ]
    )
    cov_m = np.diag(vol_ann / np.sqrt(12)) @ corr @ np.diag(vol_ann / np.sqrt(12))
    L = np.linalg.cholesky(cov_m)
    mu_m = mu_ann / 12

    shocks = rng.standard_normal((n_months, len(tickers)))
    log_ret = mu_m + shocks @ L.T
    prices = pd.DataFrame(
        np.exp(np.cumsum(log_ret, axis=0)) * 100, index=dates, columns=tickers
    )
    return prices
