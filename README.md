# Behavioral Portfolio Optimization
### Markowitz Meets Kahneman & Tversky

*A quantitative research project prepared for my Behavorial Finance course*

---

## Overview

This project implements and compares **two paradigms of portfolio optimization**:

1. **Classical Mean-Variance (Markowitz, 1952)** — the rational investor maximises Sharpe ratio
2. **Prospect Theory (Kahneman & Tversky, 1979)** — the behavioral investor maximises expected prospect value with asymmetric loss aversion

The core research questions are:

> *What is the long-run performance cost of loss aversion?*
> *Does the Myopic Loss Aversion effect (Thaler et al., 1997) hold in a diversified multi-asset setting?*

These questions sit at the heart of Amundi's Investor Intelligence research agenda (Bianchi & Brière, WP150, 2024).

---

## Results Preview

| Portfolio | Median Wealth (×W₀, 20y) | Sharpe | P(ruin) |
|-----------|--------------------------|--------|---------|
| Max Sharpe (rational) | ~×4.5 | ~1.20 | 0% |
| Prospect Theory (λ=2.25) | ~×3.1 | ~0.99 | 0% |
| Min Variance | ~×1.4 | ~0.11 | 0% |

**Behavioral bias costs ~30% of terminal wealth** over 20 years relative to the rational benchmark.

---

## Project Structure

```
portfolio_behavioral/
├── main.py                  ← run this
├── src/
│   ├── data.py              ← data loading (yfinance + synthetic fallback)
│   ├── optimization.py      ← Markowitz + Prospect Theory optimizers
│   ├── simulation.py        ← Monte Carlo engine + risk metrics
│   └── visualization.py     ← all figures
├── outputs/                 ← auto-generated figures + CSV
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Asset Universe

Seven assets covering major risk premia:

| Ticker | Asset Class |
|--------|-------------|
| SPY | US Equities |
| AGG | US Bonds (Aggregate) |
| GLD | Gold |
| VNQ | US Real Estate (REIT) |
| TLT | Long-Term US Treasuries |
| EEM | Emerging Market Equities |
| BIL | Cash / T-Bills |

Monthly data (2010–2025) from Yahoo Finance. Monthly frequency is preferred to avoid the Epps (1979) effect on daily cross-correlations.

### 2. Mean-Variance Optimization

Tangency portfolio maximises the Sharpe ratio:

$$\max_{w} \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}} \quad \text{s.t.} \quad \mathbf{1}^\top w = 1,\ w \geq 0$$

The efficient frontier is traced by solving the dual QP for 60 target-return levels.

### 3. Prospect Theory Value Function

The Kahneman-Tversky (1979) value function replaces expected utility:

$$v(x) = \begin{cases} x^\alpha & \text{if } x \geq 0 \\ -\lambda \cdot |x|^\alpha & \text{if } x < 0 \end{cases}$$

With empirically calibrated parameters (Tversky & Kahneman, 1992):  
- **λ = 2.25** — losses hurt ~2.25× more than equivalent gains  
- **α = 0.88** — diminishing sensitivity in both domains

The optimal portfolio maximises $\mathbb{E}[v(W_T - W_0)]$ estimated by Monte Carlo over a 1-year evaluation horizon.

**Implementation detail**: shocks are pre-generated once (Common Random Numbers technique, Glasserman 2003) to produce a smooth, differentiable objective for SLSQP convergence.

### 4. Monte Carlo Simulation

Terminal wealth after T=20 years is simulated as:

$$W_T = W_0 \prod_{t=1}^{T \cdot 12} (1 + r_{p,t}), \quad r_{p,t} = w^\top r_t, \quad r_t \sim \mathcal{N}(\mu_m, \Sigma_m)$$

with 10,000 paths per portfolio. Cholesky decomposition preserves the empirical correlation structure.

### 5. Risk Metrics

- **CVaR (Expected Shortfall)**: average return in the worst 5% scenarios — a coherent risk measure (Artzner et al., 1999)
- **P(ruin)**: probability of ending with less than initial capital
- **Median CAGR**: geometric mean return from the median terminal wealth

### 6. Lambda Sensitivity Analysis

A grid of λ ∈ [1, 5] traces the *behavioral efficient frontier*: how allocation and Sharpe ratio evolve as loss aversion increases. **Panel C** reveals the Myopic Loss Aversion Paradox: increasing λ improves short-run loss avoidance but can *worsen* long-run CVaR due to reinvestment risk from under-allocating to equity.

---

## What Is Original in This Project?

This project is **applied research**, not fundamental discovery. Markowitz (1952), Kahneman & Tversky (1979), and Thaler et al. (1997) are all 20th-century results. The contribution here is:

1. **Empirical combination**: integrating Prospect Theory *directly into the optimizer* (not just describing it) with real multi-asset data (2010–2025 including COVID, rate cycle).

2. **The lambda sensitivity + CVaR panel (Fig. 3, Panel C)**: quantifying the *paradox* that higher loss aversion does not reduce long-run CVaR — a result relevant for the design of investor nudges and default options.

3. **Myopic Loss Aversion in a multi-asset setting**: Thaler et al. (1997) used a simplified two-asset setup. The replication here uses 7 assets and a full covariance structure, making the results more directly applicable to wealth management practice.

4. **Direct link to robo-advice research**: the welfare cost of behavioral bias (~30% terminal wealth) provides a quantitative rationale for the type of robo-advisory intervention studied by Bianchi & Brière (WP150, 2024).

---

## Installation

```bash
git clone https://github.com/your-username/portfolio-behavioral
cd portfolio-behavioral
pip install -r requirements.txt
python main.py
```

---

## References

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77–91.
- Kahneman, D. & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk. *Econometrica*, 47(2), 263–291.
- Tversky, A. & Kahneman, D. (1992). Advances in Prospect Theory. *Journal of Risk and Uncertainty*, 5(4), 297–323.
- Thaler, R., Tversky, A., Kahneman, D. & Schwartz, A. (1997). The Effect of Myopia and Loss Aversion on Risk Taking. *Quarterly Journal of Economics*, 112(2), 647–661.
- Artzner, P., Delbaen, F., Eber, J. & Heath, D. (1999). Coherent Measures of Risk. *Mathematical Finance*, 9(3), 203–228.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- Bianchi, M. & Brière, M. (2024). Human-Robot Interactions in Investment Decisions. *Amundi Working Paper* 150.
- Epps, T.W. (1979). Comovements in Stock Prices in the Very Short Run. *Journal of the American Statistical Association*, 74(366), 291–298.

---

*Author: Thomas FANGET — Paris II Panthéon Assas — 2025-2026*  
*Contact: thomas@fanget.net*
