"""
backtester.py  —  Position Sizing, Execution Lag, and P&L
==========================================================
Component 3 of the regime_strats pipeline (Phase 4).

Responsibilities
----------------
1. Execution lag        : position[t] = f(signal[t-1], vol[t-1])
2. EWM volatility scaling : scale positions to target annualised vol
3. Transaction costs    : charge cost_bps on every unit of position change
4. Benchmark            : always-long eigenportfolio at the same target vol

Timing convention (0-indexed, array length T)
---------------------------------------------
All signals and vol estimates are computed at the close of day t and used
to size the position entered at the open of day t+1:

    ewm_vol[t]      incorporates returns[0 : t] inclusive  ← close of t
    signal[t]       uses delta_ar[t]                        ← close of t
    position[t]     = signal[t-1] * target_vol / ewm_vol[t-1]
                    = 0 at t=0 (no position before first signal)
    gross_ret[t]    = position[t] * pc_return[t]
    cost[t]         = |position[t] - position[t-1]| * cost_bps / 10_000
    net_ret[t]      = gross_ret[t] - cost[t]

No future information enters any position at time t. The shift of both
signal and vol by exactly 1 bar is the single most important invariant in
this file — do not remove it.

Volatility estimator
--------------------
Zero-mean EWM variance (RiskMetrics convention) with halflife =
eigenportfolio.halflife_primary (63d). Consistent with covariance_engine.py.
Annualised: vol[t] = sqrt(ewm_var[t] * ann_factor).
A minimum vol floor of 1e-4 (1bp) prevents division by near-zero.

Benchmark
---------
Always-long eigenportfolio at target vol, with identical vol scaling and
cost charging. Natural comparison: the regime filter's value-add is
net_ret_strategy - net_ret_bmark, not vs zero.

Output schema (backtest_df columns)
-------------------------------------
    date               — passthrough from signal_df
    selected_pc_ret    — raw eigenportfolio return (no sizing), passthrough
    in_regime          — boolean regime flag, passthrough
    ewm_vol            — annualised EWM vol at each bar (diagnostic)
    For label in ["A", "B", "C", "bmark"]:
        position_{X}   — vol-scaled, lagged, leverage-capped position
        gross_ret_{X}  — position × pc_return
        cost_{X}       — one-way transaction cost on position change (>= 0)
        net_ret_{X}    — gross_ret - cost
"""

import numpy as np
import polars as pl
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .regime_classifier import PCSelectionResult


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """
    Resolved backtesting parameters.

    Attributes
    ----------
    target_vol   : annualised vol target for position sizing (e.g. 0.10 = 10%).
    cost_bps     : one-way transaction cost in basis points (e.g. 5 = 0.05%).
    max_leverage : cap on vol-scaled position (e.g. 2.0 → max 2× notional).
                   Prevents runaway leverage when realised vol is very low.
    vol_halflife : EWM halflife for vol estimation — must match halflife_primary.
    ann_factor   : trading days per year (252).
    strategies   : list of strategy labels to backtest.
    """
    target_vol:   float
    cost_bps:     float
    max_leverage: float
    vol_halflife: int
    ann_factor:   float
    strategies: List[str] = field(default_factory=list)

    @classmethod
    def from_cfg(cls, rs_cfg: dict, ep_cfg: dict) -> "BacktestConfig":
        auto_strategies = ["A", "B", "C"]

        return cls(
            target_vol   = float(rs_cfg["target_vol"]),
            cost_bps     = float(rs_cfg["cost_bps"]),
            max_leverage = float(rs_cfg.get("max_leverage", 2.0)),
            vol_halflife = int(ep_cfg["halflife_primary"]),
            ann_factor   = float(ep_cfg.get("ann_factor", 252.0)),
            strategies   = rs_cfg.get("strategies", auto_strategies),
        )


# ---------------------------------------------------------------------------
# EWM volatility estimator
# ---------------------------------------------------------------------------

def _compute_ewm_vol(
    returns:     np.ndarray,
    halflife:    int,
    ann_factor:  float = 252.0,
    vol_floor:   float = 1e-4,
) -> np.ndarray:
    """
    Zero-mean EWM annualised volatility (RiskMetrics convention).

    Update rule : ewm_var[t] = lam * ewm_var[t-1] + (1-lam) * r[t]^2
    Annualised  : ewm_vol[t] = sqrt(ewm_var[t] * ann_factor)

    ewm_vol[t] incorporates return[t]. When used for position sizing the
    caller shifts by 1: lagged_vol[t] = ewm_vol[t-1], so position[t]
    uses only returns[0 : t-1] inclusive. See module timing convention.

    Parameters
    ----------
    returns    : (T,) daily returns of the selected eigenportfolio.
    halflife   : EWM halflife in trading days.
    ann_factor : trading days per year.
    vol_floor  : minimum annualised vol to prevent division by near-zero.

    Returns
    -------
    ewm_vol : (T,) annualised EWM vol. ewm_vol[0] seeded from r[0]^2.
    """
    lam     = 2.0 ** (-1.0 / halflife)
    T       = len(returns)
    ewm_var = np.empty(T)

    ewm_var[0] = returns[0] ** 2
    for t in range(1, T):
        ewm_var[t] = lam * ewm_var[t - 1] + (1.0 - lam) * returns[t] ** 2

    return np.maximum(np.sqrt(ewm_var * ann_factor), vol_floor)


# ---------------------------------------------------------------------------
# Position builder
# ---------------------------------------------------------------------------

def _build_positions(
    signal:     np.ndarray,
    ewm_vol:    np.ndarray,
    cfg:        BacktestConfig,
) -> np.ndarray:
    """
    Apply execution lag and vol scaling to a raw 0/1 signal.

    Steps
    -----
    1. Lag signal by 1 bar: lagged_signal[t] = signal[t-1], [0] = 0.
    2. Lag vol by 1 bar  : lagged_vol[t]    = ewm_vol[t-1], [0] = ewm_vol[0].
    3. Raw position       = lagged_signal * (target_vol / lagged_vol).
    4. Clip to [0, max_leverage] — long-only, leverage-capped.

    Parameters
    ----------
    signal  : (T,) float array, 0.0 or 1.0, from signal_generator.py.
    ewm_vol : (T,) annualised EWM vol from _compute_ewm_vol().
    cfg     : BacktestConfig.

    Returns
    -------
    position : (T,) float array of vol-scaled, lag-adjusted positions.
    """
    lagged_signal = np.concatenate([[0.0],    signal[:-1]])
    lagged_vol    = np.concatenate([[ewm_vol[0]], ewm_vol[:-1]])

    raw_position = lagged_signal * (cfg.target_vol / lagged_vol)
    return np.clip(raw_position, 0.0, cfg.max_leverage)


# ---------------------------------------------------------------------------
# Cost calculator
# ---------------------------------------------------------------------------

def _compute_costs(
    position: np.ndarray,
    cost_bps: float,
) -> np.ndarray:
    """
    One-way transaction cost on every unit of position change.

    cost[t] = |position[t] - position[t-1]| * cost_bps / 10_000

    cost[0] accounts for opening the initial position from flat.
    Result is always non-negative.

    Parameters
    ----------
    position : (T,) float array of vol-scaled positions.
    cost_bps : one-way cost in basis points.

    Returns
    -------
    cost : (T,) float array, >= 0 at every bar.
    """
    turnover = np.abs(np.diff(position, prepend=0.0))
    return turnover * cost_bps / 10_000.0


# ---------------------------------------------------------------------------
# Single-strategy backtest
# ---------------------------------------------------------------------------

def _backtest_one(
    signal:     np.ndarray,
    pc_returns: np.ndarray,
    ewm_vol:    np.ndarray,
    cfg:        BacktestConfig,
) -> Dict[str, np.ndarray]:
    """
    Run the full backtest for one strategy signal.

    Returns
    -------
    dict with keys: position, gross_ret, cost, net_ret — each (T,) float array.
    """
    position  = _build_positions(signal, ewm_vol, cfg)
    gross_ret = position * pc_returns
    cost      = _compute_costs(position, cfg.cost_bps)
    net_ret   = gross_ret - cost

    return {
        "position":  position,
        "gross_ret": gross_ret,
        "cost":      cost,
        "net_ret":   net_ret,
    }


# ---------------------------------------------------------------------------
# Diagnostic printer
# ---------------------------------------------------------------------------

def print_backtest_summary(
    backtest_df: pl.DataFrame,
    split_masks: dict,
    strategies:  List[str],
) -> None:
    """
    Print per-strategy, per-split turnover and quick-look SR to stdout.

    Reports gross SR (pre-cost) and net SR (post-cost) side-by-side so
    cost drag is immediately visible. A large gross-vs-net gap indicates
    either high turnover or cost_bps is too high for the strategy.
    """
    sep = "=" * 72
    print(f"\n{sep}")
    print("  BACKTEST SUMMARY  (positions vol-scaled to target)")
    print(sep)

    all_labels   = strategies + ["bmark"]
    split_labels = ["train", "val", "test", "full"]
    full_mask    = np.ones(len(backtest_df), dtype=bool)
    masks_ext    = {**split_masks, "full": full_mask}

    for label in all_labels:
        pos_col  = f"position_{label}"
        grss_col = f"gross_ret_{label}"
        net_col  = f"net_ret_{label}"
        cost_col = f"cost_{label}"

        if pos_col not in backtest_df.columns:
            continue

        print(f"\n  {'Benchmark (always long)' if label == 'bmark' else 'Strategy ' + label}")
        header = (
            f"    {'Split':<8}  {'Gross SR':>9}  {'Net SR':>9}  "
            f"{'Ann Ret%':>9}  {'Avg Pos':>8}  {'Ann Turnov':>11}"
        )
        print(header)
        print(f"    {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*11}")

        for split in split_labels:
            mask = masks_ext[split]
            if len(mask) != len(backtest_df):
                print(f"    {split:<8}  mask length mismatch")
                continue

            pos  = backtest_df[pos_col].to_numpy()[mask]
            gr   = backtest_df[grss_col].to_numpy()[mask]
            nr   = backtest_df[net_col].to_numpy()[mask]
            cost = backtest_df[cost_col].to_numpy()[mask]

            ann  = float(backtest_df["ewm_vol"].to_numpy()[0])   # proxy for ann_factor
            # Compute stats — avoid division by zero
            def _sr(rets):
                s = float(np.std(rets, ddof=1))
                return float(np.mean(rets)) / s * np.sqrt(252) if s > 1e-12 else np.nan

            gross_sr = _sr(gr)
            net_sr   = _sr(nr)
            ann_ret  = float(np.mean(nr)) * 252 * 100
            avg_pos  = float(np.mean(pos))
            ann_turn = float(np.sum(np.abs(np.diff(pos, prepend=0.0)))) / (len(pos) / 252)

            print(
                f"    {split:<8}  "
                f"{gross_sr:>+9.3f}  "
                f"{net_sr:>+9.3f}  "
                f"{ann_ret:>+9.2f}  "
                f"{avg_pos:>8.3f}  "
                f"{ann_turn:>10.2f}x"
            )

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_backtester(
    signal_df:   pl.DataFrame,
    split_masks: dict,
    result:      PCSelectionResult,
    rs_cfg:      dict,
    ep_cfg:      dict,
) -> pl.DataFrame:
    """
    Full backtest pipeline: vol scale → lag → cost → P&L → benchmark.

    Parameters
    ----------
    signal_df   : output of run_signal_generator() from signal_generator.py.
                  Must contain selected_pc_ret and signal_{A,B,C} columns.
    split_masks : dict from get_split_masks(panel["date"].to_list(), cfg["splits"]).
                  Must be aligned to signal_df dates (panel dates, not full df).
    result      : PCSelectionResult — for reference / logging only.
    rs_cfg      : cfg["regime_strats"].
    ep_cfg      : cfg["eigenportfolio"] — provides halflife_primary, ann_factor.

    Returns
    -------
    backtest_df : signal_df with position, gross_ret, cost, net_ret columns
                  appended for each strategy and the benchmark.
                  Primary input to performance_evaluator.py.
    """
    cfg        = BacktestConfig.from_cfg(rs_cfg, ep_cfg)
    pc_returns = signal_df["selected_pc_ret"].to_numpy().astype(float)
    strategies = cfg.strategies

    print(f"  Backtesting strategies : {strategies} + benchmark")
    print(f"    Target vol     : {cfg.target_vol:.1%}")
    print(f"    Cost           : {cfg.cost_bps:.0f} bps one-way")
    print(f"    Max leverage   : {cfg.max_leverage:.1f}x")
    print(f"    Vol halflife   : {cfg.vol_halflife}d")

    ewm_vol = _compute_ewm_vol(pc_returns, cfg.vol_halflife, cfg.ann_factor)

    new_cols: Dict[str, List] = {"ewm_vol": ewm_vol.tolist()}

    # ---- Regime strategies -------------------------------------------------
    for strat in strategies:
        col = f"signal_{strat}"
        if col not in signal_df.columns:
            print(f"  WARNING: {col} not found in signal_df — skipping strategy {strat}")
            continue
        raw_signal = signal_df[col].to_numpy().astype(float)
        result_d   = _backtest_one(raw_signal, pc_returns, ewm_vol, cfg)
        for metric, arr in result_d.items():
            new_cols[f"{metric}_{strat}"] = arr.tolist()

    # ---- Benchmark: always long at target vol ------------------------------
    bmark_signal         = np.ones(len(pc_returns), dtype=float)
    bmark_d              = _backtest_one(bmark_signal, pc_returns, ewm_vol, cfg)
    for metric, arr in bmark_d.items():
        new_cols[f"{metric}_bmark"] = arr.tolist()

    backtest_df = signal_df.with_columns([
        pl.Series(name, values)
        for name, values in new_cols.items()
    ])

    print_backtest_summary(backtest_df, split_masks, strategies)

    return backtest_df
