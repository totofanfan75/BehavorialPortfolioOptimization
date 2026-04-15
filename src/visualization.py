"""
visualization.py : Figures for the project.

All figures are designed to be self-contained (no global state) and are saved
to an output directory..
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from simulation import SimulationResult, risk_metrics

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#f9f9f9",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)

PORTFOLIO_COLORS = {
    "Max Sharpe": "#c0392b",
    "Min Variance": "#27ae60",
    "Prospect Theory": "#2980b9",
}

ASSET_COLORS = sns.color_palette("tab10", 7)


def _save(fig: plt.Figure, path: Path, filename: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / filename, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path / filename}")


# ---------------------------------------------------------------------------
# Figure 1 — Correlation matrix + Efficient frontier
# ---------------------------------------------------------------------------

def plot_frontier(
    returns: pd.DataFrame,
    mu: pd.Series,
    Sigma: pd.DataFrame,
    eff_vols: np.ndarray,
    eff_rets: np.ndarray,
    portfolios: dict,          # label → Portfolio object
    asset_labels: dict,
    out_dir: Path,
) -> plt.Figure:
    """Correlation heatmap (left) and mean-variance efficient frontier (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: correlation heatmap ---
    ax = axes[0]
    corr = returns.corr()
    sns.heatmap(
        corr,
        annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.4,
        ax=ax, cbar_kws={"shrink": 0.75},
    )
    ax.set_title("Monthly Return Correlation Matrix (2010–2025)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # --- Right: efficient frontier ---
    ax = axes[1]
    ax.plot(eff_vols, eff_rets, "k--", lw=2, label="Efficient Frontier", zorder=3)

    for i, ticker in enumerate(returns.columns):
        v = float(np.sqrt(Sigma.loc[ticker, ticker]))
        r = float(mu.loc[ticker])
        ax.scatter(v, r, s=90, color=ASSET_COLORS[i], zorder=5)
        ax.annotate(
            asset_labels.get(ticker, ticker),
            (v, r), textcoords="offset points",
            xytext=(7, 3), fontsize=9,
        )

    markers = {"Max Sharpe": ("*", "crimson", 320), "Min Variance": ("X", "seagreen", 180)}
    for label, p in portfolios.items():
        if label in markers:
            m, c, s = markers[label]
            sr = p.sharpe
            ax.scatter(
                p.volatility, p.expected_return,
                marker=m, color=c, s=s, zorder=6,
                label=f"{label}  (SR = {sr:.2f})",
            )

    ax.set_xlabel("Annualised Volatility")
    ax.set_ylabel("Annualised Expected Return")
    ax.set_title("Markowitz Efficient Frontier (Long-Only, Monthly Data)")
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "fig1_efficient_frontier.png")
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Monte Carlo distributions + fan chart
# ---------------------------------------------------------------------------

def plot_montecarlo(
    sim_results: dict[str, SimulationResult],
    out_dir: Path,
) -> plt.Figure:
    """Terminal wealth distributions (KDE) and trajectory fan charts."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # KDE of terminal wealth
    ax = axes[0]
    p97 = max(np.percentile(sr.W_final, 97) for sr in sim_results.values())
    for label, sr in sim_results.items():
        color = PORTFOLIO_COLORS.get(label.split("\n")[0], "gray")
        sns.kdeplot(sr.W_final, ax=ax, color=color, lw=2,
                    label=label, clip=(0, p97 * 1.05))
        med = np.median(sr.W_final)
        ax.axvline(med, color=color, ls="--", lw=1.3, alpha=0.8)
        q5 = np.percentile(sr.W_final, 5)
        ax.axvline(q5, color=color, ls=":", lw=1.0, alpha=0.6)

    ax.set_xlim(0, p97)
    ax.set_xlabel("Terminal Wealth Multiplier (W₀ = 1)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Terminal Wealth Distribution after {next(iter(sim_results.values())).T_years} Years\n"
        "(Dashed = median  |  Dotted = 5th percentile)"
    )
    ax.legend(fontsize=9)

    # Fan chart — median + 10th–90th percentile band
    ax = axes[1]
    for label, sr in sim_results.items():
        color = PORTFOLIO_COLORS.get(label.split("\n")[0], "gray")
        t = sr.time_axis()
        med = np.median(sr.trajectories, axis=1)
        p10 = np.percentile(sr.trajectories, 10, axis=1)
        p90 = np.percentile(sr.trajectories, 90, axis=1)
        ax.fill_between(t, p10, p90, alpha=0.12, color=color)
        ax.plot(t, med, color=color, lw=2.5,
                label=f"{label}  (median ×{np.median(sr.W_final):.2f})")

    ax.set_yscale("log")
    ax.set_xlabel("Years")
    ax.set_ylabel("Wealth (log scale)")
    ax.set_title("Simulated Wealth Paths — Median ± 10th–90th Percentile Band")
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "fig2_montecarlo.png")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Lambda sensitivity (the core research contribution)
# ---------------------------------------------------------------------------

def plot_lambda_sensitivity(
    alloc_df: pd.DataFrame,
    sharpes: list[float],
    cvars: list[float],
    lambdas: np.ndarray,
    sharpe_rational: float,
    out_dir: Path,
) -> plt.Figure:
    """Three-panel sensitivity analysis for loss aversion parameter lambda."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A — Allocation stacked area
    ax = axes[0]
    alloc_df.plot.area(ax=ax, colormap="tab10", alpha=0.82)
    ax.axvline(2.25, color="black", ls="--", lw=1.8, alpha=0.7, label="K&T empirical λ=2.25")
    ax.set_xlabel("Loss Aversion λ")
    ax.set_ylabel("Portfolio Weight")
    ax.set_title("Panel A — Optimal Allocation vs. λ\n(λ=1 rational  →  λ=5 highly loss-averse)")
    ax.legend(loc="upper right", fontsize=7)

    # Panel B — Sharpe cost
    ax = axes[1]
    ax.plot(lambdas, sharpes, "o-", color="#c0392b", lw=2, ms=5, label="Behavioral portfolio")
    ax.axhline(sharpe_rational, color="navy", ls="--", lw=1.8, label="Max Sharpe (rational)")
    ax.axvline(2.25, color="gray", ls=":", lw=1.2)
    ax.set_xlabel("Loss Aversion λ")
    ax.set_ylabel("Ex-ante Sharpe Ratio")
    ax.set_title("Panel B — Performance Cost of Loss Aversion")
    ax.legend(fontsize=9)
    # Annotate the cost at lambda=2.25
    idx_225 = np.argmin(np.abs(lambdas - 2.25))
    cost = sharpe_rational - sharpes[idx_225]
    ax.annotate(
        f"SR cost ≈ {cost:.3f}",
        xy=(2.25, sharpes[idx_225]),
        xytext=(3.0, sharpes[idx_225] + 0.05),
        fontsize=9, arrowprops=dict(arrowstyle="->", color="black"),
    )

    # Panel C — Myopic Loss Aversion paradox
    # A higher lambda does not necessarily improve long-run CVaR because
    # the investor under-allocates to equity and faces reinvestment risk.
    ax = axes[2]
    ax.plot(lambdas, [c * 100 for c in cvars], "s-", color="#e67e22", lw=2, ms=5)
    ax.axvline(2.25, color="gray", ls=":", lw=1.2)
    ax.set_xlabel("Loss Aversion λ")
    ax.set_ylabel("CVaR 5% over 20 years (%)")
    ax.set_title("Panel C — Myopic Loss Aversion Paradox\nLong-Run CVaR does not improve with λ")

    plt.suptitle(
        "Sensitivity Analysis: Impact of Loss Aversion λ on Portfolio Choice\n"
        "and the Long-Run Cost of Behavioral Biases",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    _save(fig, out_dir, "fig3_lambda_sensitivity.png")
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Myopic Loss Aversion (Thaler et al. 1997)
# ---------------------------------------------------------------------------

def plot_myopic_loss_aversion(
    alloc_df: pd.DataFrame,
    horizons: list[int],
    equity_tickers: list[str],
    rational_equity_share: float,
    out_dir: Path,
) -> plt.Figure:
    """Replicate Thaler et al. (1997): equity share collapses at short horizons."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Stacked area
    ax = axes[0]
    alloc_df.plot.area(ax=ax, colormap="tab10", alpha=0.82)
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"{h}m" for h in horizons])
    ax.set_xlabel("Evaluation Horizon")
    ax.set_ylabel("Portfolio Allocation")
    ax.set_title("Optimal Allocation by Evaluation Frequency\n(λ = 2.25, Thaler et al. 1997)")
    ax.legend(loc="upper right", fontsize=8)

    # Equity share line
    ax = axes[1]
    eq_share = alloc_df[equity_tickers].sum(axis=1) * 100
    ax.plot(range(len(horizons)), eq_share.values, "o-",
            color="#c0392b", lw=2.5, ms=9, zorder=4)
    ax.axhline(rational_equity_share * 100, color="navy", ls="--", lw=1.8,
               label="Rational equity share (Max Sharpe)")
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"{h}m" for h in horizons])
    ax.set_xlabel("Evaluation Horizon (months)")
    ax.set_ylabel("Total Equity Allocation (%)")
    ax.set_title("Myopic Loss Aversion Effect\nEquity share vs. evaluation frequency")
    ax.legend(fontsize=9)

    # Annotations
    ax.annotate(
        "Reviews portfolio\nmonthly\n→ avoids equities",
        xy=(0, float(eq_share.iloc[0])),
        xytext=(0.8, float(eq_share.iloc[0]) - 12),
        fontsize=9, arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax.annotate(
        "Reviews every\n5 years\n→ holds more equity",
        xy=(len(horizons) - 1, float(eq_share.iloc[-1])),
        xytext=(len(horizons) - 2.8, float(eq_share.iloc[-1]) + 8),
        fontsize=9, arrowprops=dict(arrowstyle="->", color="black"),
    )

    plt.tight_layout()
    _save(fig, out_dir, "fig4_myopic_loss_aversion.png")
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Final comparison dashboard
# ---------------------------------------------------------------------------

def plot_dashboard(
    portfolios: dict,
    sim_results: dict[str, SimulationResult],
    returns: pd.DataFrame,
    out_dir: Path,
) -> plt.Figure:
    """Comprehensive dashboard comparing rational vs behavioral portfolios."""
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35)

    names = list(portfolios.keys())
    colors = [PORTFOLIO_COLORS.get(n.split("\n")[0], "steelblue") for n in names]

    # Panel 1 — Allocations
    ax0 = fig.add_subplot(gs[0, 0])
    alloc = pd.DataFrame(
        {n: p.weights * 100 for n, p in portfolios.items()},
        index=returns.columns,
    )
    alloc.T.plot(kind="bar", ax=ax0, colormap="tab10", width=0.65, edgecolor="none")
    ax0.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax0.set_ylabel("Weight (%)")
    ax0.set_title("Portfolio Allocations")
    ax0.legend(fontsize=7, loc="upper right")

    # Panel 2 — KDE
    ax1 = fig.add_subplot(gs[0, 1])
    T = next(iter(sim_results.values())).T_years
    p97 = max(np.percentile(sr.W_final, 97) for sr in sim_results.values())
    for (lbl, sr), col in zip(sim_results.items(), colors):
        sns.kdeplot(sr.W_final, ax=ax1, color=col, lw=2, label=lbl, clip=(0, p97 * 1.05))
        ax1.axvline(np.median(sr.W_final), color=col, ls="--", lw=1.1, alpha=0.7)
    ax1.set_xlim(0, p97)
    ax1.set_xlabel("Terminal Wealth (×W₀)")
    ax1.set_title(f"Terminal Wealth Distribution — {T} years")
    ax1.legend(fontsize=8)

    # Panel 3 — Metrics table
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    rows = []
    for lbl, sr in sim_results.items():
        m = risk_metrics(sr)
        rows.append([lbl, m["Median (x W0)"], m["CAGR (median)"],
                     m[f"CVaR 95%"], m["P(ruin)"], m["P(x3)"]])
    tbl = ax2.table(
        cellText=rows,
        colLabels=["Portfolio", "Median", "CAGR", "CVaR 95%", "P(ruin)", "P(×3)"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ecf0f1")
    ax2.set_title("Risk / Return Metrics (20-Year Horizon)", pad=35)

    # Panel 4 — Wealth trajectories (full width)
    ax3 = fig.add_subplot(gs[1, :])
    for (lbl, sr), col in zip(sim_results.items(), colors):
        t = sr.time_axis()
        med = np.median(sr.trajectories, axis=1)
        p10 = np.percentile(sr.trajectories, 10, axis=1)
        p90 = np.percentile(sr.trajectories, 90, axis=1)
        ax3.fill_between(t, p10, p90, alpha=0.10, color=col)
        ax3.plot(t, med, color=col, lw=2.5,
                 label=f"{lbl}  (median ×{np.median(sr.W_final):.2f})")

    ax3.set_yscale("log")
    ax3.set_xlabel("Years")
    ax3.set_ylabel("Wealth (log scale)")
    ax3.set_title(f"Simulated Wealth Trajectories — {T} Years Horizon\nMedian ± [10th, 90th] percentile band")
    ax3.legend(fontsize=9)

    plt.suptitle(
        "Portfolio Comparison Dashboard — Rational vs. Behavioral Allocation\n"
        "Markowitz (1952)  ·  Min Variance  ·  Kahneman–Tversky Prospect Theory",
        fontsize=13, y=1.01,
    )
    _save(fig, out_dir, "fig5_dashboard.png")
    return fig
