import os
from datetime import datetime, date as date_type
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HALF_LIFE_LABELS: Dict[int, str] = {
    21:  "HL=21d (1 mo)",
    63:  "HL=63d (1 qtr)",
    125: "HL=125d (6 mo)",
    252: "HL=252d (1 yr)",
}

MARKET_EVENTS = {
    "2008-09-15": "Lehman",
    "2010-05-06": "Flash\nCrash",
    "2011-08-05": "S&P\nDowngrade",
    "2015-08-24": "China\nCrash",
    "2018-02-05": "VIX\nShock",
    "2020-03-16": "COVID",
    "2022-01-03": "Rate\nHikes",
}

# Base palettes — extended beyond 4 so adding half-lives never raises IndexError
_BASE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

REGIME_COLORS = ["#2ca02c", "#ff7f0e", "#d62728"]   # Low / Elevated / Stress


# ---------------------------------------------------------------------------
# Dynamic style generators  (replaces hardcoded PC_COLORS / PC_LABELS / COLORS)
# ---------------------------------------------------------------------------

def _get_pc_styles(K: int):
    """
    Return (labels, colors) for K principal components.
    Works for any K — no hardcoded limit of 3.
    """
    cmap   = cm.get_cmap("tab10")
    labels = [f"PC{k + 1}" for k in range(K)]
    colors = [mcolors.to_hex(cmap(k % 10)) for k in range(K)]
    return labels, colors


def _get_hl_colors(n: int) -> List[str]:
    """Return n colours for half-life series, extending beyond 4 if needed."""
    if n <= len(_BASE_COLORS):
        return _BASE_COLORS[:n]
    return [mcolors.to_hex(cm.get_cmap("tab10")(i % 10)) for i in range(n)]


def _hl_label(hl: int) -> str:
    """Safe half-life label lookup — falls back to 'HL={hl}d' for unknown values."""
    return HALF_LIFE_LABELS.get(hl, f"HL={hl}d")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_event_lines(ax: plt.Axes, ymax: float) -> None:
    for date_str, label in MARKET_EVENTS.items():
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        ax.axvline(dt, color="grey", lw=0.8, ls="--", alpha=0.5)
        ax.text(dt, ymax * 0.97, label, fontsize=5.5, color="grey",
                ha="center", va="top")


def _to_datetime(d) -> datetime:
    """Coerce a polars date / datetime.date / string to datetime.datetime."""
    if isinstance(d, datetime):
        return d
    if isinstance(d, date_type):
        return datetime(d.year, d.month, d.day)
    return datetime.strptime(str(d)[:10], "%Y-%m-%d")


def _to_mpl_dates(dates_list: List) -> np.ndarray:
    """Convert a list of heterogeneous date objects to matplotlib date floats."""
    return np.array([mdates.date2num(_to_datetime(d)) for d in dates_list])


def _add_event_lines_mpl(ax: plt.Axes, ymin: float, ymax: float) -> None:
    """Event lines for axes using numeric (mpl) date x-axis (pcolormesh panels)."""
    for date_str, label in MARKET_EVENTS.items():
        dt  = datetime.strptime(date_str, "%Y-%m-%d")
        num = mdates.date2num(dt)
        ax.axvline(num, color="grey", lw=0.8, ls="--", alpha=0.5)
        ax.text(num, ymax - 0.08 * (ymax - ymin), label,
                fontsize=5.5, color="grey", ha="center", va="top")


# ---------------------------------------------------------------------------
# Fig 1 — Rolling absorption ratio
# ---------------------------------------------------------------------------

def plot_absorption_ratios(
    dates: List,
    decomp_results: Dict[int, Dict],
    output_dir: str = "results",
) -> None:
    """Fig 1 — Rolling absorption ratio for all half-lives."""
    sorted_hls = sorted(decomp_results.keys())
    hl_colors  = _get_hl_colors(len(sorted_hls))

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, hl in enumerate(sorted_hls):
        warmup = int(decomp_results[hl]["warmup_steps"])
        ax.plot(dates[warmup:], decomp_results[hl]["absorption_ratios"],
                label=_hl_label(hl), color=hl_colors[i], lw=1.2)
    _, ymax = ax.get_ylim()
    _add_event_lines(ax, ymax)
    ax.set_title("Rolling Absorption Ratio (fraction of variance in top-K PCs)", fontsize=12)
    ax.set_ylabel("Absorption Ratio")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig1_absorption_ratio.png"), dpi=150)
    plt.close(fig)
    print("Saved fig1_absorption_ratio.png")


# ---------------------------------------------------------------------------
# Fig 2 — Standardised delta-AR
# ---------------------------------------------------------------------------

def plot_delta_ar(
    dates: List,
    monitor_results: Dict[int, Dict],
    decomp_results: Dict[int, Dict],
    output_dir: str = "results",
) -> None:
    """Fig 2 — Standardised delta-AR signal for all half-lives."""
    sorted_hls = sorted(monitor_results.keys())
    n_hl       = len(sorted_hls)
    hl_colors  = _get_hl_colors(n_hl)

    fig, axes = plt.subplots(n_hl, 1, figsize=(14, 3 * n_hl), sharex=True)
    if n_hl == 1:
        axes = [axes]

    for i, hl in enumerate(sorted_hls):
        warmup = int(decomp_results[hl]["warmup_steps"])
        ax     = axes[i]
        ax.plot(dates[warmup:], monitor_results[hl]["delta_ar"],
                color=hl_colors[i], lw=0.9)
        ax.axhline(0,  color="black", lw=0.6)
        ax.axhline(1,  color="red",   lw=0.6, ls=":", alpha=0.7, label="+1σ")
        ax.axhline(-1, color="green", lw=0.6, ls=":", alpha=0.7, label="-1σ")
        _, ymax = ax.get_ylim()
        _add_event_lines(ax, ymax)
        ax.set_ylabel("ΔAR (σ)", fontsize=8)
        ax.set_title(_hl_label(hl), fontsize=9, loc="left")
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    fig.suptitle(
        "Standardised Absorption Ratio Shift — ΔAR = (AR_short − AR_long) / σ_long",
        fontsize=11,
    )
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_delta_ar.png"), dpi=150)
    plt.close(fig)
    print("Saved fig2_delta_ar.png")


# ---------------------------------------------------------------------------
# Fig 3 — Champion vs Challenger
# ---------------------------------------------------------------------------

def plot_champion_vs_challenger(
    dates: List,
    monitor_results: Dict[int, Dict],
    decomp_results: Dict[int, Dict],
    output_dir: str = "results",
) -> None:
    """Fig 3 — PCA absorption ratio vs equal-weight baseline per half-life."""
    sorted_hls = sorted(monitor_results.keys())
    n_hl       = len(sorted_hls)
    hl_colors  = _get_hl_colors(n_hl)
    ncols      = min(2, n_hl)
    nrows      = (n_hl + ncols - 1) // ncols

    fig, axes_grid = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 4.5 * nrows),
        sharex=True, sharey=True,
    )
    axes_flat = np.array(axes_grid).flatten()

    for i, hl in enumerate(sorted_hls):
        ax      = axes_flat[i]
        warmup  = int(decomp_results[hl]["warmup_steps"])
        d       = monitor_results[hl]
        t_dates = dates[warmup:]
        ax.plot(t_dates, d["pca_explained"],
                label="PCA (Challenger)",        color=hl_colors[i], lw=1.2)
        ax.plot(t_dates, d["ew_explained"],
                label="Equal-Weight (Baseline)", color="grey", lw=1.0, ls="--")
        ax.fill_between(
            t_dates, d["ew_explained"], d["pca_explained"],
            where=d["pca_explained"] > d["ew_explained"],
            alpha=0.15, color=hl_colors[i], label="PCA edge",
        )
        _, ymax = ax.get_ylim()
        _add_event_lines(ax, ymax)
        ax.set_title(_hl_label(hl), fontsize=9)
        ax.legend(fontsize=7)
        ax.set_ylabel("Fraction of variance explained")

    # Hide unused subplot panels when n_hl < nrows * ncols
    for j in range(n_hl, len(axes_flat)):
        axes_flat[j].set_visible(False)

    for ax in axes_flat[(nrows - 1) * ncols : (nrows - 1) * ncols + ncols]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(3))

    fig.suptitle("Champion vs Challenger: PCA vs Equal-Weight Factor", fontsize=12)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_champion_vs_challenger.png"), dpi=150)
    plt.close(fig)
    print("Saved fig3_champion_vs_challenger.png")


# ---------------------------------------------------------------------------
# Fig 4 — Dynamic K retained
# ---------------------------------------------------------------------------

def plot_num_components(
    dates: List,
    decomp_results: Dict[int, Dict],
    output_dir: str = "results",
) -> None:
    """Fig 4 — Dynamic number of PCs retained over time."""
    sorted_hls = sorted(decomp_results.keys())
    hl_colors  = _get_hl_colors(len(sorted_hls))

    fig, ax = plt.subplots(figsize=(14, 4))
    for i, hl in enumerate(sorted_hls):
        warmup = int(decomp_results[hl]["warmup_steps"])
        ax.plot(dates[warmup:], decomp_results[hl]["num_components"],
                label=_hl_label(hl), color=hl_colors[i], lw=1.0,
                drawstyle="steps-post")
    _, ymax = ax.get_ylim()
    _add_event_lines(ax, ymax)
    ax.set_title("Dynamic Number of PCs Retained (variance_threshold from config.yaml)",
                 fontsize=12)
    ax.set_ylabel("K (number of PCs)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_num_components.png"), dpi=150)
    plt.close(fig)
    print("Saved fig4_num_components.png")


# ---------------------------------------------------------------------------
# Fig 5 — Sector Loadings Heatmap  (Phase 1)
# ---------------------------------------------------------------------------

def plot_sector_loadings_heatmap(
    dates: List,
    decomp_results: Dict[int, Dict],
    tickers: List[str],
    primary_hl: int,
    num_components: int,
    output_dir: str = "results",
) -> None:
    """
    Fig 5 — Rolling sector loadings heatmap for the top-K PCs.

    K rows (one per PC), N sectors on Y-axis, time on X-axis.
    Works for any K and any N — no hardcoded assumptions.
    """
    from eigenportfolio import get_sector_loadings_timeseries

    warmup      = int(decomp_results[primary_hl]["warmup_steps"])
    loadings_ts = get_sector_loadings_timeseries(
        decomp_results, primary_hl, num_components
    )  # (T_cov, N, K)

    T_cov     = loadings_ts.shape[0]
    N         = len(tickers)
    K         = loadings_ts.shape[2]   # actual K from data, not parameter
    t_dates   = dates[warmup : warmup + T_cov]
    x_mpl     = _to_mpl_dates(t_dates)
    y_centers = np.arange(N)

    vmax = np.percentile(np.abs(loadings_ts), 95) * 1.05
    vmax = max(vmax, 0.05)

    fig, axes = plt.subplots(K, 1, figsize=(15, 3.2 * K), sharex=True)
    if K == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        data = loadings_ts[:, :, k].T   # (N, T_cov)
        im   = ax.pcolormesh(
            x_mpl, y_centers, data,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            shading="nearest",
        )
        ax.set_yticks(np.arange(N))
        ax.set_yticklabels(tickers, fontsize=8)
        ax.set_ylabel("Sector", fontsize=8)
        ax.set_title(
            f"PC{k + 1} — rolling sector eigenvector loading  (HL={primary_hl}d)",
            fontsize=9, loc="left",
        )
        _add_event_lines_mpl(ax, -0.5, N - 0.5)
        cbar = fig.colorbar(im, ax=ax, pad=0.01, aspect=25)
        cbar.set_label("Loading", fontsize=7)
        cbar.ax.tick_params(labelsize=7)

    axes[-1].xaxis_date()
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].set_xlabel("Date")

    fig.suptitle(
        f"Fig 5 — Rolling Sector Loadings Heatmap  (HL={primary_hl}d)\n"
        "Red = sector moves with factor | Blue = sector moves against factor",
        fontsize=11,
    )
    fig.autofmt_xdate()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig5_sector_loadings_heatmap.png"), dpi=150)
    plt.close(fig)
    print("Saved fig5_sector_loadings_heatmap.png")


# ---------------------------------------------------------------------------
# Fig 6 — Eigenportfolio P&L  (Phase 1)
# ---------------------------------------------------------------------------

def plot_eigenportfolio_pnl(
    dates: List,
    ep_results: Dict[int, Dict],
    ep_perf_all: Dict[int, Dict],
    primary_hl: int,
    output_dir: str = "results",
) -> None:
    """
    Fig 6 — Cumulative P&L + rolling Sharpe for all K eigenportfolios.

    Top panel  : growth of $1.
    Bottom panel: rolling 252-day Sharpe ratio.
    K is derived from ep_returns.shape[1] — no hardcoded limit of 3.
    """
    ep   = ep_results[primary_hl]
    perf = ep_perf_all[primary_hl]

    offset    = ep["dates_offset"]
    T_eff     = ep["ep_returns"].shape[0]
    K         = ep["ep_returns"].shape[1]   # derived from data
    t_dates   = dates[offset : offset + T_eff]
    cum_ret   = perf["cumulative_returns"]
    wealth    = cum_ret + 1.0
    pc_labels, pc_colors = _get_pc_styles(K)

    window      = 252
    roll_sharpe = np.full((T_eff, K), np.nan)
    for t in range(window, T_eff):
        r_slice = ep["ep_returns"][t - window : t, :]
        mu  = np.mean(r_slice, axis=0) * 252.0
        sig = np.std(r_slice,  axis=0, ddof=1) * np.sqrt(252.0)
        roll_sharpe[t, :] = np.where(sig > 1e-12, mu / sig, 0.0)

    fig, (ax_pnl, ax_sr) = plt.subplots(
        2, 1, figsize=(14, 9),
        sharex=True, gridspec_kw={"height_ratios": [2, 1]},
    )

    for k in range(K):
        sharpe_full = float(perf["sharpe"][k])
        ann_ret     = float(perf["annualised_return"][k]) * 100.0
        max_dd      = float(perf["max_drawdown"][k]) * 100.0
        label = (
            f"{pc_labels[k]}  |  SR={sharpe_full:+.2f}  "
            f"Ann={ann_ret:+.1f}%  MDD={max_dd:.1f}%"
        )
        ax_pnl.plot(t_dates, wealth[:, k], label=label,
                    color=pc_colors[k], lw=1.3)

    ax_pnl.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
    _, ymax_pnl = ax_pnl.get_ylim()
    _add_event_lines(ax_pnl, ymax_pnl)
    ax_pnl.set_ylabel("Growth of $1  (log-return eigenportfolios)")
    ax_pnl.set_title(
        f"Fig 6 — Eigenportfolio Cumulative P&L  (HL={primary_hl}d, L2-unit weights)",
        fontsize=12,
    )
    ax_pnl.legend(loc="upper left", fontsize=8)
    ax_pnl.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    for k in range(K):
        ax_sr.plot(t_dates, roll_sharpe[:, k], label=pc_labels[k],
                   color=pc_colors[k], lw=0.9, alpha=0.85)
    ax_sr.axhline(0, color="black", lw=0.6)
    ax_sr.axhline(1, color="grey",  lw=0.6, ls=":", alpha=0.6)
    _, ymax_sr = ax_sr.get_ylim()
    _add_event_lines(ax_sr, ymax_sr)
    ax_sr.set_ylabel("Rolling 252d Sharpe")
    ax_sr.legend(loc="upper left", fontsize=8, ncol=min(K, 4))
    ax_sr.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_sr.xaxis.set_major_locator(mdates.YearLocator(2))

    fig.autofmt_xdate()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig6_eigenportfolio_pnl.png"), dpi=150)
    plt.close(fig)
    print("Saved fig6_eigenportfolio_pnl.png")


# ---------------------------------------------------------------------------
# Fig 7 — Delta-AR vs Forward VIX  (Phase 3)
# ---------------------------------------------------------------------------

def plot_vix_predictive(
    predictive_results: dict,
    output_dir: str = "results",
) -> None:
    """
    Fig 7 — ΔAR vs Forward VIX predictive regression.

    Layout
    ------
    Top row    : one scatter per forward horizon h (driven by len(ols.keys())).
    Bottom row : rolling OLS R² time series for all horizons on one panel.

    All counts (n_h, n_regimes) are derived from the data — no hardcoding.
    """
    da        = predictive_results["delta_ar"]
    vix       = predictive_results["vix_levels"]
    regime    = predictive_results["vix_regime"]
    labels    = predictive_results["regime_labels"]
    ols       = predictive_results["ols"]
    roll_r2   = predictive_results["rolling_r2"]
    horizons  = sorted(ols.keys())
    n_h       = len(horizons)
    n_regimes = len(labels)

    # Generate regime colours dynamically for any number of bins
    r_cmap    = cm.get_cmap("RdYlGn_r")
    reg_colors = [mcolors.to_hex(r_cmap(i / max(n_regimes - 1, 1)))
                  for i in range(n_regimes)]

    hl_colors = _get_hl_colors(n_h)

    fig = plt.figure(figsize=(5 * n_h, 11))
    gs  = fig.add_gridspec(
        2, n_h,
        height_ratios=[1.6, 1],
        hspace=0.45,
        wspace=0.30,
    )

    # ── Top row: scatter per horizon ─────────────────────────────────────────
    for col, h in enumerate(horizons):
        ax  = fig.add_subplot(gs[0, col])
        res = ols[h]
        X_h   = da[:-h]
        Y_h   = vix[h:]
        reg_h = regime[:-h]

        for r_idx, (r_label, r_color) in enumerate(zip(labels, reg_colors)):
            mask = reg_h == r_idx
            ax.scatter(X_h[mask], Y_h[mask],
                       c=r_color, s=4, alpha=0.35, linewidths=0, label=r_label)

        ax.plot(res["x_fit"], res["y_fit"],
                color="black", lw=1.5, ls="--", label="OLS fit")

        sig = "**" if res["p_value"] < 0.01 else ("*" if res["p_value"] < 0.05 else "")
        ax.set_title(
            f"h = {h}d  |  R² = {res['r2']:.4f}{sig}  β = {res['slope']:+.3f}",
            fontsize=9,
        )
        ax.set_xlabel("ΔAR (σ)", fontsize=8)
        ax.set_ylabel(f"VIX_{{t+{h}}}", fontsize=8)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.legend(fontsize=6, markerscale=2, loc="upper left")

    # ── Bottom row: rolling R² time series ───────────────────────────────────
    ax_roll = fig.add_subplot(gs[1, :])

    for h, color in zip(horizons, hl_colors):
        h_dates, r2_arr = roll_r2[h]
        ax_roll.plot(
            [_to_datetime(d) for d in h_dates], r2_arr,
            color=color, lw=0.9, alpha=0.85, label=f"h={h}d",
        )

    ax_roll.set_ylim(bottom=0)
    _, ymax_roll = ax_roll.get_ylim()
    _add_event_lines(ax_roll, max(ymax_roll, 0.05))
    ax_roll.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_roll.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_roll.set_ylabel("Rolling R² (252d window)", fontsize=8)
    ax_roll.set_xlabel("Date", fontsize=8)
    ax_roll.set_title("Rolling OLS R²: ΔAR_t → VIX_{t+h}", fontsize=9, loc="left")
    ax_roll.legend(fontsize=8, loc="upper right")
    ax_roll.tick_params(labelsize=7)

    fig.suptitle(
        "Fig 7 — ΔAR vs Forward VIX  "
        "(ΔAR is concurrent with volatility, not a leading predictor)\n"
        "** p<0.01  * p<0.05  (autocorrelation inflates significance)",
        fontsize=11,
    )
    fig.autofmt_xdate()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig7_vix_predictive.png"), dpi=150)
    plt.close(fig)
    print("Saved fig7_vix_predictive.png")


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def run_all(
    dates: List,
    decomp_results: Dict[int, Dict],
    monitor_results: Dict[int, Dict],
    output_dir: str = "results",
    ep_results:         Optional[Dict[int, Dict]] = None,
    ep_perf_all:        Optional[Dict[int, Dict]] = None,
    tickers:            Optional[List[str]]       = None,
    ep_config:          Optional[Dict]            = None,
    predictive_results: Optional[Dict]            = None,
) -> None:
    """
    Generate all figures.

    Figs 1-4 always run (Phase 0 core).
    Figs 5-6 run when ep_results is provided (Phase 1+).
    Fig  7   runs when predictive_results is provided (Phase 3+).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Phase 0 ───────────────────────────────────────────────────────────────
    plot_absorption_ratios(dates, decomp_results, output_dir)
    plot_delta_ar(dates, monitor_results, decomp_results, output_dir)
    plot_champion_vs_challenger(dates, monitor_results, decomp_results, output_dir)
    plot_num_components(dates, decomp_results, output_dir)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if ep_results is not None and ep_perf_all is not None:
        if tickers is None:
            raise ValueError("tickers must be provided alongside ep_results for Figs 5/6.")
        if not ep_config:
            raise ValueError("ep_config must be provided with halflife_primary and num_components.")
        primary_hl = int(ep_config["halflife_primary"])
        num_comp   = int(ep_config["num_components"])

        plot_sector_loadings_heatmap(
            dates, decomp_results, tickers,
            primary_hl=primary_hl,
            num_components=num_comp,
            output_dir=output_dir,
        )
        plot_eigenportfolio_pnl(
            dates, ep_results, ep_perf_all,
            primary_hl=primary_hl,
            output_dir=output_dir,
        )

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    if predictive_results is not None:
        plot_vix_predictive(predictive_results, output_dir)

    print(f"All figures saved to ./{output_dir}/")
