"""
regime_visualizer.py  —  Phase 4 Figures
=========================================
Three figures summarising the regime strategy pipeline results.

Fig 8 : Regime Timeline        — ΔAR signal, Strategy A position, and PC2
                                  eigenportfolio equity curve with regime
                                  shading and market event lines.
Fig 9 : Strategy Equity Curves — Cumulative P&L and drawdown for all
                                  strategies vs benchmark with split
                                  boundaries and per-split SR annotations.
Fig 10: SR by Split             — Grouped bar chart comparing all strategies
                                  across train / val / test splits.
                                  Error bars = Lo (2002) i.i.d. SR SE.

Style convention: matches visualizer.py
    figsize=(14, X), dpi=150, COLORS, MARKET_EVENTS, _add_event_lines.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Dict, List

from .visualizer          import MARKET_EVENTS, _BASE_COLORS as COLORS, _add_event_lines
from .regime_classifier   import PCSelectionResult
from .performance_evaluator import PerformanceMetrics


# ---------------------------------------------------------------------------
# Display metadata
# ---------------------------------------------------------------------------

STRAT_LABELS = {
    "A":     "Strategy A (time exit)",
    "B":     "Strategy B (signal exit)",
    "C":     "Strategy C (combination)",
    "bmark": "Benchmark (always long)",
}
STRAT_COLORS = {
    "A":     "#1f77b4",   # blue    — matches COLORS[0]
    "B":     "#ff7f0e",   # orange  — matches COLORS[1]
    "C":     "#9467bd",   # purple
    "bmark": "#7f7f7f",   # grey
}
REGIME_COLOR = "#4CAF50"    # green shading — diversifying regime
SPLIT_COLOR  = "#444444"    # dark grey — train/val/test dividers


# ---------------------------------------------------------------------------
# Shared layout helpers
# ---------------------------------------------------------------------------

def _parse_dates(backtest_df) -> List[datetime]:
    """Convert backtest_df date strings to datetime objects."""
    return [datetime.strptime(d, "%Y-%m-%d")
            for d in backtest_df["date"].to_list()]


def _split_boundaries(
    dates_dt:    List[datetime],
    split_masks: dict,
) -> Dict[str, datetime]:
    """Last date of train and val splits for axvline placement."""
    train_idx = int(np.where(split_masks["train"])[0][-1])
    val_idx   = int(np.where(split_masks["val"])[0][-1])
    return {
        "train_end": dates_dt[train_idx],
        "val_end":   dates_dt[val_idx],
    }


def _split_midpoints(
    dates_dt:    List[datetime],
    split_masks: dict,
) -> Dict[str, datetime]:
    """Midpoint date of each split for annotation placement."""
    result = {}
    for name in ["train", "val", "test"]:
        idx = np.where(split_masks[name])[0]
        if len(idx) > 0:
            result[name] = dates_dt[int(idx[len(idx) // 2])]
    return result


def _add_split_lines(
    ax:         plt.Axes,
    boundaries: Dict[str, datetime],
) -> None:
    """Dotted vertical dividers at train|val and val|test boundaries."""
    for label, dt in [("Train | Val", boundaries["train_end"]),
                      ("Val | Test",  boundaries["val_end"])]:
        ax.axvline(dt, color=SPLIT_COLOR, lw=1.1, ls=":", alpha=0.7, zorder=5)
        ax.text(
            dt, 0.98, label,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=6.5,
            color=SPLIT_COLOR, rotation=90, zorder=6,
        )


def _shade_regime(
    ax:        plt.Axes,
    dates_dt:  List[datetime],
    in_regime: np.ndarray,
) -> None:
    """Light green shading across the full axes height when in_regime=True."""
    ax.fill_between(
        dates_dt, 0, 1,
        where=in_regime,
        transform=ax.get_xaxis_transform(),
        color=REGIME_COLOR, alpha=0.13, zorder=0,
        label="Diversifying Regime (in-position)",
    )


def _fmt_xaxis(ax: plt.Axes, every_n: int = 2) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(every_n))


def _compute_drawdown_pct(cum_ret: np.ndarray) -> np.ndarray:
    """Drawdown series in % (negative) from a wealth-index array."""
    peak = np.maximum.accumulate(cum_ret)
    return (cum_ret - peak) / np.maximum(peak, 1e-12) * 100.0


# ---------------------------------------------------------------------------
# Figure 8 — Regime Timeline
# ---------------------------------------------------------------------------

def plot_regime_timeline(
    backtest_df:  "pl.DataFrame",
    cum_ret_dict: Dict[str, np.ndarray],
    split_masks:  dict,
    result:       PCSelectionResult,
    rs_cfg:       dict,
    output_dir:   str = "results",
) -> None:
    """
    Fig 8 — Three-panel regime timeline.

    Panel 1 : ΔAR signal with entry/exit threshold lines.
              Green zones mark diversifying regime periods.
              Market event lines (Lehman, COVID, etc.) for visual audit.
    Panel 2 : Strategy A vol-scaled position over time.
              Shows exact exposure carried day-to-day.
    Panel 3 : PC2 eigenportfolio cumulative return (benchmark — no filter).
              Confirms the selected factor appreciates during regime periods,
              validating the conditioning logic visually.
    """
    dates_dt   = _parse_dates(backtest_df)
    in_regime  = backtest_df["in_regime"].to_numpy().astype(bool)
    delta_ar   = backtest_df["delta_ar"].to_numpy().astype(float)
    position_a = backtest_df["position_A"].to_numpy().astype(float)
    pc_label   = f"PC{result.best_pc + 1}"
    entry_thr  = float(result.entry_threshold)
    exit_thr   = float(rs_cfg["exit_threshold_signal"])
    boundaries = _split_boundaries(dates_dt, split_masks)

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [4, 2, 2]},
    )

    # ── Panel 1: ΔAR ────────────────────────────────────────────────────────
    ax = axes[0]
    _shade_regime(ax, dates_dt, in_regime)
    ax.plot(dates_dt, delta_ar, color=COLORS[1], lw=0.85, zorder=3, label="ΔAR")
    ax.axhline(0,         color="black",   lw=0.5, alpha=0.35)
    ax.axhline(entry_thr, color=COLORS[0], lw=1.2, ls="--",
               label=f"Entry {entry_thr:+.1f}σ")
    ax.axhline(exit_thr,  color=COLORS[2], lw=1.0, ls=":",
               label=f"Exit (B/C) {exit_thr:+.1f}σ")
    ax.axhline(+1.0,      color=COLORS[3], lw=0.8, ls=":", alpha=0.55,
               label="+1.0σ fragile")
    _, ymax = ax.get_ylim()
    _add_event_lines(ax, ymax)
    _add_split_lines(ax, boundaries)
    ax.set_ylabel("ΔAR (σ)", fontsize=9)
    ax.set_title(
        f"ΔAR Signal  ·  Entry: {entry_thr:+.1f}σ  ·  Exit (B/C): {exit_thr:+.1f}σ",
        fontsize=10, loc="left",
    )
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)

    # ── Panel 2: Strategy A position ────────────────────────────────────────
    ax = axes[1]
    _shade_regime(ax, dates_dt, in_regime)
    ax.fill_between(dates_dt, 0, position_a,
                    color=COLORS[0], alpha=0.55, zorder=3)
    ax.plot(dates_dt, position_a, color=COLORS[0], lw=0.7, zorder=4)
    _add_split_lines(ax, boundaries)
    ax.set_ylabel("Position\n(vol-scaled)", fontsize=8)
    ax.set_title("Strategy A — Vol-Scaled Position", fontsize=10, loc="left")
    ax.set_ylim(bottom=0)

    # ── Panel 3: Selected PC cumulative return ───────────────────────────────
    ax = axes[2]
    bmark_cum = cum_ret_dict.get("bmark")
    if bmark_cum is not None:
        _shade_regime(ax, dates_dt, in_regime)
        ax.plot(dates_dt, bmark_cum, color=COLORS[3], lw=0.85, zorder=3,
                label=f"{pc_label} eigenportfolio (benchmark)")
        ax.axhline(1.0, color="black", lw=0.5, alpha=0.35)
        _add_split_lines(ax, boundaries)
        ax.set_ylabel("Wealth\nIndex", fontsize=8)
        ax.set_title(
            f"{pc_label} Eigenportfolio (benchmark — no regime filter)",
            fontsize=10, loc="left",
        )
        ax.legend(loc="upper left", fontsize=7.5)

    _fmt_xaxis(axes[-1])
    fig.suptitle(
        f"Fig 8 — Regime Timeline: ΔAR Signal · Strategy A Position · "
        f"{pc_label} Eigenportfolio",
        fontsize=11,
    )
    fig.autofmt_xdate()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig8_regime_timeline.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   Saved fig8_regime_timeline.png")


# ---------------------------------------------------------------------------
# Figure 9 — Equity Curves + Drawdown
# ---------------------------------------------------------------------------

def plot_equity_curves(
    backtest_df:  "pl.DataFrame",
    cum_ret_dict: Dict[str, np.ndarray],
    perf_results: Dict[str, Dict[str, PerformanceMetrics]],
    split_masks:  dict,
    strategies:   List[str],
    output_dir:   str = "results",
) -> None:
    """
    Fig 9 — Equity curves (top) and Strategy A drawdown vs benchmark (bottom).

    Top panel annotations: per-split SR and MDD for Strategy A in small
    text boxes at the base of each split region — makes train/val/test
    performance immediately readable without needing to consult the table.

    Bottom panel: the -35% benchmark MDD vs -11% Strategy A MDD is the
    single most visually compelling result in the whole project.
    """
    dates_dt   = _parse_dates(backtest_df)
    in_regime  = backtest_df["in_regime"].to_numpy().astype(bool)
    boundaries = _split_boundaries(dates_dt, split_masks)
    midpoints  = _split_midpoints(dates_dt, split_masks)

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # ── Panel 1: Equity curves ───────────────────────────────────────────────
    ax = axes[0]
    _shade_regime(ax, dates_dt, in_regime)

    # Benchmark first (muted, behind strategies)
    if "bmark" in cum_ret_dict:
        ax.plot(dates_dt, cum_ret_dict["bmark"],
                color=STRAT_COLORS["bmark"], lw=1.1, ls="--", alpha=0.65,
                label=STRAT_LABELS["bmark"], zorder=3)

    for strat in strategies:
        if strat not in cum_ret_dict:
            continue
        ax.plot(
            dates_dt, cum_ret_dict[strat],
            color=STRAT_COLORS[strat],
            lw=1.9 if strat == "A" else 1.0,
            alpha=1.0 if strat == "A" else 0.6,
            label=STRAT_LABELS[strat],
            zorder=4 if strat == "A" else 3,
        )

    ax.axhline(1.0, color="black", lw=0.5, alpha=0.35)
    _, ymax = ax.get_ylim()
    _add_event_lines(ax, ymax * 0.88)
    _add_split_lines(ax, boundaries)

    # Per-split SR + MDD annotations for Strategy A
    box_style = dict(
        boxstyle="round,pad=0.3", facecolor="white",
        alpha=0.88, edgecolor=STRAT_COLORS["A"], lw=0.8,
    )
    for split_name, mid_dt in midpoints.items():
        pm = perf_results.get("A", {}).get(split_name)
        if pm is None:
            continue
        oos_flag = "" if split_name == "train" else " (OOS)"
        note = (f"{split_name.capitalize()}{oos_flag}\n"
                f"SR = {pm.annualised_sr:+.2f}\n"
                f"MDD = {pm.max_drawdown_pct:.1f}%")
        ax.text(
            mid_dt, 0.04, note,
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=7.0,
            color=STRAT_COLORS["A"], bbox=box_style, zorder=7,
        )

    ax.set_ylabel("Wealth Index (start = 1.0)", fontsize=9)
    ax.set_title("Strategy Equity Curves — Net of 5bps One-Way Cost  "
                 "· Green = Diversifying Regime", fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5)

    # ── Panel 2: Drawdown ────────────────────────────────────────────────────
    ax = axes[1]
    _shade_regime(ax, dates_dt, in_regime)

    if "A" in cum_ret_dict:
        dd_a = _compute_drawdown_pct(cum_ret_dict["A"])
        ax.fill_between(dates_dt, 0, dd_a,
                        color=STRAT_COLORS["A"], alpha=0.4, zorder=3)
        ax.plot(dates_dt, dd_a, color=STRAT_COLORS["A"], lw=0.9,
                label=f"{STRAT_LABELS['A']}  (max {dd_a.min():.1f}%)",
                zorder=4)

    if "bmark" in cum_ret_dict:
        dd_b = _compute_drawdown_pct(cum_ret_dict["bmark"])
        ax.plot(dates_dt, dd_b, color=STRAT_COLORS["bmark"], lw=1.1,
                ls="--", alpha=0.75,
                label=f"{STRAT_LABELS['bmark']}  (max {dd_b.min():.1f}%)",
                zorder=3)

    ax.axhline(0, color="black", lw=0.5, alpha=0.35)
    _add_split_lines(ax, boundaries)
    ax.set_ylabel("Drawdown (%)", fontsize=9)
    ax.set_title("Drawdown: Strategy A vs Benchmark", fontsize=10, loc="left")
    ax.legend(loc="lower left", fontsize=8.5)

    _fmt_xaxis(axes[-1])
    fig.suptitle(
        "Fig 9 — Phase 4: Regime Strategy Equity Curves  ·  "
        "PC2 Eigenportfolio + ΔAR Signal",
        fontsize=11,
    )
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig9_equity_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   Saved fig9_equity_curves.png")


# ---------------------------------------------------------------------------
# Figure 10 — SR by Split (grouped bar chart)
# ---------------------------------------------------------------------------

def plot_sr_by_split(
    perf_results: Dict[str, Dict[str, PerformanceMetrics]],
    strategies:   List[str],
    cost_bps:     float,
    output_dir:   str = "results",
) -> None:
    """
    Fig 10 — Annualised net SR grouped by split, all strategies + benchmark.

    Error bars = Lo (2002) i.i.d. SE — a lower bound (see module docstring
    of performance_evaluator.py for the autocorrelation caveat).

    Val underperformance is economically expected — 2015–2018 was a calm
    low-vol bull market where ΔAR rarely fell below -1.0σ. The test SR
    (2019–2026) is the honest held-out OOS result.
    """
    splits       = ["train", "val", "test"]
    split_titles = [
        "Train\n(2005–2014)\n[in-sample]",
        "Val\n(2015–2018)\n[OOS]",
        "Test\n(2019–2026)\n[OOS — held-out]",
    ]
    all_labels = strategies + ["bmark"]
    n_bars     = len(all_labels)
    bar_w      = 0.17
    group_gap  = 0.25
    x_centers  = np.arange(len(splits)) * (n_bars * bar_w + group_gap)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, label in enumerate(all_labels):
        srs, ses = [], []
        for split in splits:
            pm  = perf_results.get(label, {}).get(split)
            srs.append(pm.annualised_sr if pm else float("nan"))
            ses.append(pm.sr_stderr     if pm else float("nan"))

        x_pos = x_centers + i * bar_w

        ax.bar(x_pos, srs, width=bar_w * 0.88,
               color=STRAT_COLORS[label], alpha=0.85,
               label=STRAT_LABELS[label], zorder=3)

        # Error bars (SE — lower bound)
        valid = [(x, sr, se) for x, sr, se in zip(x_pos, srs, ses)
                 if not (np.isnan(sr) or np.isnan(se))]
        if valid:
            xs, ys, es = zip(*valid)
            ax.errorbar(xs, ys, yerr=es, fmt="none",
                        ecolor="black", elinewidth=0.9,
                        capsize=3, capthick=0.9, zorder=5)

        # Value labels
        for x, sr in zip(x_pos, srs):
            if not np.isnan(sr):
                v_off = 0.02 if sr >= 0 else -0.04
                va    = "bottom" if sr >= 0 else "top"
                ax.text(x, sr + v_off, f"{sr:+.2f}",
                        ha="center", va=va, fontsize=6.5, zorder=6)

    ax.axhline(0, color="black", lw=0.8, alpha=0.5)

    # Val underperformance callout
    val_center = x_centers[1] + (n_bars - 1) * bar_w / 2
    ax.text(val_center, ax.get_ylim()[0] * 0.72,
            "Low-vol bull (2015–18):\nregime rarely fired →\nstrategies mostly flat",
            ha="center", va="top", fontsize=7.0,
            color="#888", style="italic", zorder=7)

    ax.set_xticks(x_centers + (n_bars - 1) * bar_w / 2)
    ax.set_xticklabels(split_titles, fontsize=9)
    ax.set_ylabel(f"Annualised SR (net of {cost_bps:.0f}bps one-way)", fontsize=10)
    ax.set_title(
        "Fig 10 — Net Sharpe Ratio by Split  ·  "
        "Error bars = Lo (2002) i.i.d. SE  [lower bound — see notes]",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="y", lw=0.5, alpha=0.35, zorder=0)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig10_sr_by_split.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   Saved fig10_sr_by_split.png")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_regime_visualizations(
    backtest_df:  "pl.DataFrame",
    cum_ret_dict: Dict[str, np.ndarray],
    perf_results: Dict[str, Dict[str, PerformanceMetrics]],
    split_masks:  dict,
    result:       PCSelectionResult,
    rs_cfg:       dict,
    ep_cfg:       dict,
    output_dir:   str = "results",
) -> None:
    """
    Generate all three Phase 4 figures (figs 8–10).

    Parameters
    ----------
    backtest_df  : output of run_backtester().
    cum_ret_dict : cum_ret_dict from run_performance_evaluator().
    perf_results : perf_results from run_performance_evaluator().
    split_masks  : get_split_masks(panel["date"].to_list(), cfg["splits"]).
                   Must be aligned to panel (not full df) dates.
    result       : PCSelectionResult from regime_classifier.
    rs_cfg       : cfg["regime_strats"].
    output_dir   : directory to write PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    strategies = list(rs_cfg.get("strategies", ["A", "B", "C"]))
    cost_bps   = float(rs_cfg.get("cost_bps", 5.0))

    print(f"  Generating Phase 4 figures → ./{output_dir}/")
    plot_regime_timeline(backtest_df, cum_ret_dict, split_masks,
                         result, rs_cfg, output_dir)
    plot_equity_curves(backtest_df, cum_ret_dict, perf_results,
                       split_masks, strategies, output_dir)
    plot_sr_by_split(perf_results, strategies, cost_bps, output_dir)
    print("  Phase 4 figures complete.")
