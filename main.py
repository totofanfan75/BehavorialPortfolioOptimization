"""
main.py : Entry point for the Behavioral Portfolio Optimization project.

Run:
    python main.py

All outputs (figures + metrics CSV) are written to ./outputs/.

Project structure
-----------------
    src/data.py          — data loading and preprocessing
    src/optimization.py  — Markowitz + Prospect Theory optimizers
    src/simulation.py    — Monte Carlo engine and risk metrics
    src/visualization.py — figures

Research questions addressed
-----------------------------
    1. How does the Kahneman-Tversky value function alter optimal asset allocation
       relative to the classical Markowitz tangency portfolio?
    2. What is the long-run performance cost of loss aversion (lambda sensitivity)?
    3. Does the Myopic Loss Aversion result of Thaler et al. (1997) hold in a
       multi-asset framework with real data?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from data import load_returns, TICKERS, ASSET_LABELS
from optimization import (
    max_sharpe_portfolio,
    min_variance_portfolio,
    efficient_frontier,
    prospect_theory_portfolio,
    lambda_sensitivity,
    myopic_loss_aversion,
)
from simulation import simulate, risk_metrics
from visualization import (
    plot_frontier,
    plot_montecarlo,
    plot_lambda_sensitivity,
    plot_myopic_loss_aversion,
    plot_dashboard,
)

OUT_DIR = Path("outputs")
SEED    = 42

# ===========================================================================
# 1. DATA
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 1 — Data Loading")
print("=" * 65)
returns, mu, Sigma, _ = load_returns(TICKERS, seed=SEED)
rf = float(mu["BIL"])
print(f"Risk-free rate (BIL): {rf * 100:.2f}% p.a.")
print(f"Sample: {returns.index[0].date()} → {returns.index[-1].date()}")
print(f"Observations: {len(returns)} monthly log-returns")

# ===========================================================================
# 2. MARKOWITZ OPTIMIZATION
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 2 — Markowitz Optimization")
print("=" * 65)

p_sharpe = max_sharpe_portfolio(mu, Sigma, rf)
p_minvol = min_variance_portfolio(mu, Sigma)

print(f"\n{'Portfolio':<30} {'Return':>8} {'Volatility':>12} {'Sharpe':>8}")
print("-" * 62)
for p in [p_sharpe, p_minvol]:
    print(f"{p.label:<30} {p.expected_return*100:>7.2f}%  {p.volatility*100:>10.2f}%  {p.sharpe:>7.3f}")

eff_vols, eff_rets = efficient_frontier(mu, Sigma)

# ===========================================================================
# 3. PROSPECT THEORY OPTIMIZATION
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 3 — Prospect Theory (Kahneman & Tversky, 1979)")
print("=" * 65)

# Pre-generate fixed shocks for the optimizer (1-year evaluation horizon)
# Rationale: 12 months is the standard evaluation period used in Thaler et al.
# (1997) and the typical reporting frequency for retail investors.
T_OPT_MONTHS = 12
N_SIM_OPT    = 20_000
rng_opt = np.random.default_rng(SEED)
Z_fixed = rng_opt.standard_normal((T_OPT_MONTHS, N_SIM_OPT, len(TICKERS)))

p_behav = prospect_theory_portfolio(mu, Sigma, Z_fixed, lam=2.25, alpha=0.88)
print(f"\n{'Portfolio':<45} {'Return':>8} {'Volatility':>12} {'Sharpe':>8}")
print("-" * 77)
print(f"{p_behav.label:<45} {p_behav.expected_return*100:>7.2f}%  {p_behav.volatility*100:>10.2f}%  {p_behav.sharpe:>7.3f}")

# ===========================================================================
# 4. MONTE CARLO SIMULATIONS
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 4 — Monte Carlo Simulations (20-year horizon, 10,000 paths)")
print("=" * 65)

N_SIM   = 10_000
T_YEARS = 20

portfolios = {
    "Max Sharpe":      p_sharpe,
    "Min Variance":    p_minvol,
    "Prospect Theory": p_behav,
}

sim_results = {}
for name, p in portfolios.items():
    print(f"  Simulating: {name} ...")
    sim_results[name] = simulate(p.weights, mu, Sigma, n_sim=N_SIM, T_years=T_YEARS, seed=SEED)

# Print risk table
print("\n--- Risk / Return Summary ---")
rows = []
for name, sr in sim_results.items():
    m = risk_metrics(sr)
    rows.append(m)
    print(f"\n  {name}")
    for k, v in m.items():
        print(f"    {k:<22}: {v}")

# Export to CSV
metrics_df = pd.DataFrame(rows, index=list(sim_results.keys()))
metrics_df.to_csv(OUT_DIR / "risk_metrics.csv")

# ===========================================================================
# 5. SENSITIVITY ANALYSIS - LAMBDA
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 5 — Lambda Sensitivity Analysis")
print("=" * 65)

LAMBDAS = np.round(np.linspace(1.0, 5.0, 13), 2)
alloc_lambda_df = lambda_sensitivity(mu, Sigma, Z_fixed, LAMBDAS)

# Compute ex-post Sharpe and CVaR for each lambda portfolio
sharpes, cvars = [], []
for lam, row in alloc_lambda_df.iterrows():
    sr_mc = simulate(row.values, mu, Sigma, n_sim=N_SIM, T_years=T_YEARS, seed=SEED)
    m = risk_metrics(sr_mc)
    from optimization import portfolio_stats
    ret_l, vol_l = portfolio_stats(row.values, mu, Sigma)
    sharpes.append((ret_l - rf) / vol_l)
    # Extract numeric CVaR
    cvar_str = m["CVaR 95%"]
    cvars.append(float(cvar_str.replace("%", "")) / 100)

# ===========================================================================
# 6. MYOPIC LOSS AVERSION (Thaler et al. 1997)
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 6 — Myopic Loss Aversion (Thaler et al. 1997)")
print("=" * 65)

HORIZONS = [1, 3, 6, 12, 24, 36, 60]
mla_df = myopic_loss_aversion(mu, Sigma, horizons_months=HORIZONS,
                               lam=2.25, alpha=0.88, n_sim=N_SIM_OPT, seed_base=SEED)

# ===========================================================================
# 7. FIGURES
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 7 — Generating Figures")
print("=" * 65)

plot_frontier(
    returns, mu, Sigma, eff_vols, eff_rets,
    portfolios, ASSET_LABELS, OUT_DIR,
)

plot_montecarlo(sim_results, OUT_DIR)

plot_lambda_sensitivity(
    alloc_lambda_df, sharpes, cvars, LAMBDAS,
    sharpe_rational=(p_sharpe.expected_return - rf) / p_sharpe.volatility,
    out_dir=OUT_DIR,
)

rational_eq = (
    p_sharpe.weights[list(mu.index).index("SPY")]
    + p_sharpe.weights[list(mu.index).index("EEM")]
)
plot_myopic_loss_aversion(
    mla_df, HORIZONS,
    equity_tickers=["SPY", "EEM"],
    rational_equity_share=rational_eq,
    out_dir=OUT_DIR,
)

plot_dashboard(portfolios, sim_results, returns, OUT_DIR)

# ===========================================================================
# 8. SUMMARY
# ===========================================================================
print("\n" + "=" * 65)
print("SUMMARY — Economic Interpretation")
print("=" * 65)

W_rational = np.median(sim_results["Max Sharpe"].W_final)
W_behavioral = np.median(sim_results["Prospect Theory"].W_final)
cost_pct = (W_rational - W_behavioral) / W_rational * 100

print(f"""
Median terminal wealth (20-year horizon, 10,000 simulations):

  Max Sharpe (rational)         : ×{W_rational:.2f}
  Prospect Theory (λ=2.25)      : ×{W_behavioral:.2f}

  → Wealth cost of behavioral bias: {cost_pct:.1f}% of terminal wealth
    lost relative to the rational benchmark.

Interpretation:
  The Prospect Theory investor under-allocates to equities due to loss aversion,
  which generates a significant drag on long-run compounding. This validates the
  rationale for robo-advisory tools that override short-term loss aversion and
  maintain a diversified equity exposure.

All outputs saved to: {OUT_DIR.resolve()}
""")
