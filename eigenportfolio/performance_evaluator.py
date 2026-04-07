"""
performance_evaluator.py  —  Strategy Performance Analytics
============================================================
Component 4 (final) of the regime_strats pipeline (Phase 4).

Responsibilities
----------------
1. Compute a full suite of performance metrics per strategy per split.
2. Compare each strategy against the always-long benchmark.
3. Flag SR degradation from train → val → test (overfitting indicator).
4. Export a clean summary DataFrame for CSV persistence and visualisation.

Train / val / test discipline
------------------------------
Train metrics are IN-SAMPLE. PC selection was performed on the train set,
so train SR is optimistically biased by construction — it is reported for
reference only, not for strategy evaluation.

Val is the primary evaluation set: it is out-of-sample from PC selection
but was available during development. The val SR is the correct metric
for comparing strategies A, B, C.

Test is the held-out set: never touched until final evaluation. The test
SR is the single most credible number in the whole pipeline. A meaningful
gap between val SR and test SR indicates regime shift or overfitting not
caught in val.

Sharpe ratio standard error
----------------------------
SE(SR_annual) = sqrt((ann_factor + SR_annual² / 2) / T)

This is the Lo (2002) formula under i.i.d. daily returns. Regime strategies
exhibit positive serial correlation in returns (positions are held for days
to weeks), so the i.i.d. SE underestimates the true SE. The correct
estimator uses Newey-West standard errors on the daily return series.
We report the i.i.d. SE as a lower bound — flag this explicitly in any
written-up analysis or interview presentation.

Output
------
run_performance_evaluator() returns:
  perf_results : nested dict  — perf_results[strategy][split] -> PerformanceMetrics
  summary_df   : pl.DataFrame — metrics × (strategy × split), ready for CSV export
  cum_ret_dict : dict         — cumulative return series for visualiser
"""

import numpy as np
import polars as pl
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    """
    Full performance profile for one strategy on one split.

    All return/SR metrics are computed on net_ret (after transaction costs).
    Attributes ending in _pct are already multiplied by 100 (percentage form).

    Attributes
    ----------
    annualised_sr     : annualised Sharpe ratio (net). nan if std = 0.
    sr_stderr         : Lo (2002) i.i.d. SE of SR — a lower bound (see module note).
    annualised_ret_pct: annualised net return in %.
    max_drawdown_pct  : maximum peak-to-trough drawdown in % (positive = loss).
    calmar_ratio      : annualised_ret / max_drawdown magnitude. nan if MDD = 0.
    win_rate_pct      : % of bars with strictly positive net return.
    ann_turnover      : annualised position turnover (sum of |Δpos| per year).
    pct_in_market     : % of bars with position > 0.01 (i.e. non-trivially long).
    n_obs             : number of bars in the split.
    sr_vs_bmark       : annualised_sr  − benchmark annualised_sr (same split).
    ret_vs_bmark_pct  : annualised_ret − benchmark annualised_ret (same split), %.
    """
    annualised_sr:      float
    sr_stderr:          float
    annualised_ret_pct: float
    max_drawdown_pct:   float
    calmar_ratio:       float
    win_rate_pct:       float
    ann_turnover:       float
    pct_in_market:      float
    n_obs:              int
    sr_vs_bmark:        float = field(default=float("nan"))
    ret_vs_bmark_pct:   float = field(default=float("nan"))


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def _compute_max_drawdown(returns: np.ndarray) -> float:
    """
    Maximum peak-to-trough drawdown (negative value, e.g. -0.25 = -25%).

    Computed on wealth index W[t] = prod(1 + r[i] for i in 0..t).
    Returns 0.0 if there is no drawdown (all-positive return period).
    """
    if len(returns) == 0:
        return 0.0
    cum  = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / np.maximum(peak, 1e-12)
    return float(np.min(dd))


def _compute_sr_stderr(sr_annual: float, n_obs: int, ann_factor: float) -> float:
    """
    Lo (2002) i.i.d. Sharpe ratio standard error.

    SE = sqrt((ann_factor + SR_annual² / 2) / n_obs)

    This is a lower bound for strategies with autocorrelated returns.
    Newey-West SE (not implemented here) is the correct estimator for
    those cases — flag this in written analysis.

    Returns nan if n_obs < 2 or SR is nan.
    """
    if n_obs < 2 or np.isnan(sr_annual):
        return float("nan")
    return float(np.sqrt((ann_factor + sr_annual ** 2 / 2.0) / n_obs))


def _compute_metrics(
    net_ret:    np.ndarray,
    position:   np.ndarray,
    ann_factor: float,
) -> Tuple[float, float, float, float, float, float, float, float, int]:
    """
    Compute all scalar metrics for a net return and position series.

    Returns
    -------
    (annualised_sr, sr_stderr, annualised_ret_pct, max_drawdown_pct,
     calmar_ratio, win_rate_pct, ann_turnover, pct_in_market, n_obs)
    """
    T   = len(net_ret)
    nan = float("nan")

    if T < 2:
        return (nan, nan, nan, nan, nan, nan, nan, nan, T)

    mu  = float(np.mean(net_ret))
    sig = float(np.std(net_ret, ddof=1))

    sr      = (mu / sig * np.sqrt(ann_factor)) if sig > 1e-12 else nan
    sr_se   = _compute_sr_stderr(sr, T, ann_factor)
    ann_ret = mu * ann_factor * 100.0

    mdd_raw  = _compute_max_drawdown(net_ret)
    mdd_pct  = abs(mdd_raw) * 100.0

    calmar   = (ann_ret / mdd_pct) if mdd_pct > 1e-8 else nan

    win_rate = float(np.mean(net_ret > 0)) * 100.0

    turnover    = np.abs(np.diff(position, prepend=0.0))
    years        = T / ann_factor
    ann_turn     = float(np.sum(turnover)) / years if years > 0 else nan

    pct_in = float(np.mean(position > 0.01)) * 100.0

    return (sr, sr_se, ann_ret, mdd_pct, calmar, win_rate, ann_turn, pct_in, T)


def _metrics_from_arrays(
    net_ret:    np.ndarray,
    position:   np.ndarray,
    ann_factor: float,
) -> PerformanceMetrics:
    """Thin wrapper: compute metrics and pack into PerformanceMetrics dataclass."""
    sr, sr_se, ann_ret, mdd, calmar, win, turn, pct_in, n = _compute_metrics(
        net_ret, position, ann_factor
    )
    return PerformanceMetrics(
        annualised_sr      = sr,
        sr_stderr          = sr_se,
        annualised_ret_pct = ann_ret,
        max_drawdown_pct   = mdd,
        calmar_ratio       = calmar,
        win_rate_pct       = win,
        ann_turnover       = turn,
        pct_in_market      = pct_in,
        n_obs              = n,
    )


# ---------------------------------------------------------------------------
# Per-strategy, per-split evaluation
# ---------------------------------------------------------------------------

def _evaluate_all_splits(
    backtest_df: pl.DataFrame,
    split_masks: dict,
    label:       str,
    ann_factor:  float,
) -> Dict[str, PerformanceMetrics]:
    """
    Evaluate one strategy (or "bmark") across all splits.

    Parameters
    ----------
    backtest_df : full backtest output from backtester.py.
    split_masks : dict with keys "train", "val", "test" — boolean np.ndarray,
                  length == len(backtest_df).
    label       : strategy label e.g. "A", "B", "C", "bmark".
    ann_factor  : trading days per year (252).

    Returns
    -------
    dict mapping split_name -> PerformanceMetrics (keys: train, val, test, full).
    """
    net_col = f"net_ret_{label}"
    pos_col = f"position_{label}"

    if net_col not in backtest_df.columns:
        raise KeyError(f"Column {net_col!r} not found in backtest_df. "
                       f"Available: {backtest_df.columns}")

    full_nr  = backtest_df[net_col].to_numpy().astype(float)
    full_pos = backtest_df[pos_col].to_numpy().astype(float)

    full_mask = np.ones(len(backtest_df), dtype=bool)
    masks_ext = {**split_masks, "full": full_mask}

    return {
        split: _metrics_from_arrays(full_nr[mask], full_pos[mask], ann_factor)
        for split, mask in masks_ext.items()
    }


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------

def _attach_benchmark_comparison(
    perf_results: Dict[str, Dict[str, PerformanceMetrics]],
    strategies:   List[str],
) -> None:
    """
    Fill sr_vs_bmark and ret_vs_bmark_pct for every strategy × split in-place.

    The benchmark ("bmark") is always-long eigenportfolio at target vol.
    Comparison: strategy metric − benchmark metric on the same split.
    """
    if "bmark" not in perf_results:
        return

    for strat in strategies:
        if strat not in perf_results:
            continue
        for split, m in perf_results[strat].items():
            bm = perf_results["bmark"].get(split)
            if bm is None:
                continue
            m.sr_vs_bmark      = (m.annualised_sr      - bm.annualised_sr)
            m.ret_vs_bmark_pct = (m.annualised_ret_pct - bm.annualised_ret_pct)


# ---------------------------------------------------------------------------
# Summary DataFrame builder
# ---------------------------------------------------------------------------

def _build_summary_df(
    perf_results: Dict[str, Dict[str, PerformanceMetrics]],
    all_labels:   List[str],
    splits:       List[str],
) -> pl.DataFrame:
    """
    Build a wide Polars DataFrame: metric × (strategy_split) columns.

    Column naming: f"{metric}__{label}_{split}"
    e.g. annualised_sr__A_val, max_drawdown_pct__bmark_test

    The double underscore separates the metric name from the label_split
    suffix, making it easy to split programmatically later.

    Returns
    -------
    pl.DataFrame with one row per metric, one column per strategy × split.
    """
    metric_names = [f.name for f in fields(PerformanceMetrics)]
    rows = {m: [] for m in metric_names}
    col_order = ["metric"]
    data: Dict[str, List] = {"metric": metric_names}

    for label in all_labels:
        for split in splits:
            col = f"{label}_{split}"
            col_order.append(col)
            m_dict = perf_results.get(label, {})
            pm     = m_dict.get(split)
            for mname in metric_names:
                val = getattr(pm, mname) if pm is not None else float("nan")
                data.setdefault(col, []).append(
                    float(val) if not isinstance(val, int) else int(val)
                )

    return pl.DataFrame(data).select(["metric"] + [
        c for c in col_order if c != "metric"
    ])


# ---------------------------------------------------------------------------
# Formatted report printer
# ---------------------------------------------------------------------------

def print_performance_report(
    perf_results: Dict[str, Dict[str, PerformanceMetrics]],
    strategies:   List[str],
    cost_bps:     float,
) -> None:
    """
    Print a formatted per-strategy performance table to stdout.

    Rows   : metrics
    Columns: train / val / test / full
    Footer : vs benchmark delta for SR and return

    Overfitting flag: prints a warning if SR degrades by more than 0.30
    from train to val, which may indicate parameter overfitting.
    """
    sep      = "=" * 72
    all_labs = strategies + ["bmark"]
    splits   = ["train", "val", "test", "full"]

    print(f"\n{sep}")
    print(f"  PERFORMANCE REPORT  (net of {cost_bps:.0f}bps one-way costs)")
    print(f"  *** Train = IN-SAMPLE.  Val / Test = OUT-OF-SAMPLE. ***")
    print(sep)

    metric_rows = [
        ("Ann SR",        "annualised_sr",      "{:>+9.3f}"),
        ("SE(SR) [lb]",   "sr_stderr",          "{:>9.3f} "),
        ("Ann Ret%",       "annualised_ret_pct", "{:>+9.2f}%"),
        ("Max DD%",        "max_drawdown_pct",   "{:>9.2f}%"),
        ("Calmar",         "calmar_ratio",       "{:>9.3f} "),
        ("Win Rate%",      "win_rate_pct",       "{:>9.2f}%"),
        ("Ann Turnov",     "ann_turnover",       "{:>9.2f}x"),
        ("% In Mkt",       "pct_in_market",      "{:>9.2f}%"),
        ("N Obs",          "n_obs",              "{:>9d}  "),
    ]

    for label in all_labs:
        is_bmark = label == "bmark"
        title    = "Benchmark (always long)" if is_bmark else f"Strategy {label}"
        print(f"\n  {title}")
        header = f"    {'Metric':<14}" + "".join(f"  {s.capitalize():>11}" for s in splits)
        print(header)
        print(f"    {'-'*14}" + "  " + "  ".join(["-"*11] * len(splits)))

        for row_label, attr, fmt in metric_rows:
            vals = []
            for split in splits:
                pm  = perf_results.get(label, {}).get(split)
                val = getattr(pm, attr) if pm is not None else float("nan")
                try:
                    vals.append(fmt.format(val))
                except (ValueError, TypeError):
                    vals.append(f"{'nan':>10} ")
            print(f"    {row_label:<14}" + "".join(f"  {v:>11}" for v in vals))

        # ── vs benchmark deltas ──────────────────────────────────────────────
        if not is_bmark:
            print(f"    {'─'*14}" + "  " + "  ".join(["─"*11] * len(splits)))
            for delta_label, attr in [("ΔSR vs bmark", "sr_vs_bmark"),
                                       ("ΔRet% vs bmark", "ret_vs_bmark_pct")]:
                vals = []
                for split in splits:
                    pm  = perf_results.get(label, {}).get(split)
                    val = getattr(pm, attr) if pm is not None else float("nan")
                    try:
                        vals.append(f"{val:>+10.3f} ")
                    except (ValueError, TypeError):
                        vals.append(f"{'nan':>10} ")
                print(f"    {delta_label:<14}" + "".join(f"  {v:>11}" for v in vals))

            # Overfitting flag: SR degradation train → val
            train_sr = getattr(
                perf_results.get(label, {}).get("train"), "annualised_sr", float("nan")
            )
            val_sr   = getattr(
                perf_results.get(label, {}).get("val"),   "annualised_sr", float("nan")
            )
            if not np.isnan(train_sr) and not np.isnan(val_sr):
                degradation = train_sr - val_sr
                if degradation > 0.30:
                    print(
                        f"\n    ⚠ SR degradation train→val: {degradation:+.3f}  "
                        f"(> 0.30 — possible overfitting)"
                    )

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Cumulative return series builder
# ---------------------------------------------------------------------------

def _build_cum_returns(
    backtest_df: pl.DataFrame,
    all_labels:  List[str],
) -> Dict[str, np.ndarray]:
    """
    Build wealth-index cumulative return series for each strategy.

    cum_ret[t] = prod(1 + net_ret[i] for i in 0..t), starting at 1.0.
    Used by the visualiser for equity curve plots.

    Returns
    -------
    dict mapping label -> (T,) numpy array of cumulative wealth index.
    """
    cum_dict: Dict[str, np.ndarray] = {}
    for label in all_labels:
        col = f"net_ret_{label}"
        if col not in backtest_df.columns:
            continue
        ret = backtest_df[col].to_numpy().astype(float)
        cum_dict[label] = np.cumprod(1.0 + ret)
    return cum_dict


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_performance_evaluator(
    backtest_df: pl.DataFrame,
    split_masks: dict,
    rs_cfg:      dict,
    ep_cfg:      dict,
) -> Tuple[Dict[str, Dict[str, PerformanceMetrics]], pl.DataFrame, Dict[str, np.ndarray]]:
    """
    Full performance evaluation pipeline.

    Parameters
    ----------
    backtest_df : output of run_backtester() from backtester.py.
                  Must contain net_ret_{A,B,C,bmark} and position_{A,B,C,bmark}.
    split_masks : dict from get_split_masks(panel["date"].to_list(), cfg["splits"]).
                  Must be aligned to backtest_df (panel dates, not full df dates).
    rs_cfg      : cfg["regime_strats"] — provides strategies, cost_bps.
    ep_cfg      : cfg["eigenportfolio"] — provides ann_factor.

    Returns
    -------
    perf_results : nested dict — perf_results[label][split] -> PerformanceMetrics
    summary_df   : pl.DataFrame — wide-format metrics table, ready for CSV export.
    cum_ret_dict : dict — label -> (T,) cumulative wealth index array for visualiser.

    Notes
    -----
    perf_results includes "bmark" in addition to the strategy labels.
    sr_vs_bmark and ret_vs_bmark_pct fields are populated in-place after
    all strategies are evaluated (requires bmark metrics to be ready first).
    """
    strategies = list(rs_cfg.get("strategies", ["A", "B", "C"]))
    ann_factor = float(ep_cfg.get("ann_factor", 252.0))
    cost_bps   = float(rs_cfg.get("cost_bps", 5.0))
    all_labels = strategies + ["bmark"]
    splits     = ["train", "val", "test", "full"]

    print(f"  Evaluating strategies : {strategies} + benchmark")
    print(f"    Splits              : train / val / test / full")
    print(f"    Ann factor          : {ann_factor:.0f}d")

    # ── Evaluate all strategies + benchmark ─────────────────────────────────
    perf_results: Dict[str, Dict[str, PerformanceMetrics]] = {}
    for label in all_labels:
        try:
            perf_results[label] = _evaluate_all_splits(
                backtest_df, split_masks, label, ann_factor
            )
        except KeyError as e:
            print(f"  WARNING: {e} — skipping {label}")

    # ── Attach benchmark deltas ──────────────────────────────────────────────
    _attach_benchmark_comparison(perf_results, strategies)

    # ── Print report ─────────────────────────────────────────────────────────
    print_performance_report(perf_results, strategies, cost_bps)

    # ── Build summary DataFrame ──────────────────────────────────────────────
    summary_df = _build_summary_df(perf_results, all_labels, splits)

    # ── Build cumulative return series ───────────────────────────────────────
    cum_ret_dict = _build_cum_returns(backtest_df, all_labels)

    return perf_results, summary_df, cum_ret_dict
