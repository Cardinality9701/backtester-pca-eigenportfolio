"""
signal_generator.py  —  Three-Strategy Signal Construction
===========================================================
Component 2 of the regime_strats pipeline (Phase 4).

Takes the labelled panel from regime_classifier.py and applies three exit
rules to produce per-day position signals for the backtester.

Strategies
----------
A : Hard time exit
        Enter when in_regime=True. Hold for exactly max_hold_days trading days,
        then exit regardless of delta_ar. Re-enters immediately on the next day
        if in_regime is still True.

B : Signal (trailing) exit
        Enter when in_regime=True. Hold until delta_ar recovers past
        exit_threshold_signal. Hysteresis is built-in: entry requires
        delta_ar < -1.0σ but exit only requires delta_ar > -0.5σ, so
        a position is never closed on the same bar it was opened.

C : Combination — exit on whichever condition triggers first
        Inherits entry from A and B. Exits when EITHER max_hold_days is
        reached OR the signal exit condition fires; whichever is earlier.

Signal convention
-----------------
All signals are 0.0 (flat) or +1.0 (long the selected eigenportfolio).
Short positions are never taken. PC selection guarantees that the chosen
(k, direction, threshold) combo has positive conditional SR going long;
going short is never the optimal side by construction.

Look-ahead discipline
---------------------
signal[t] is determined solely by delta_ar[t] and the current position
state — both known at the close of day t. The backtester shifts signals
forward by 1 day before constructing positions (execution at day t+1 open).
No future information is used here.

State continuity across splits
-------------------------------
The state machine runs end-to-end on the full panel without resetting at
split boundaries. A position opened at the end of the train period can
carry into val. Split masks are used only for diagnostic statistics and
performance evaluation, never for signal generation.
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
class SignalConfig:
    """
    Resolved signal parameters for all three strategies.
    Derived once from PCSelectionResult + rs_cfg at pipeline start.
    Centralises all thresholds so state machines never touch cfg directly.

    Attributes
    ----------
    entry_threshold       : ΔAR z-score for entry  — from PCSelectionResult.
    signal_direction      : +1 or -1               — from PCSelectionResult.
    exit_threshold_signal : ΔAR z-score for B/C exit — from rs_cfg.
    max_hold_days         : hard exit after N days for A/C — from rs_cfg.
    strategies            : list of strategy labels to generate.
    """
    entry_threshold:       float
    signal_direction:      int
    exit_threshold_signal: float
    max_hold_days:         int
    strategies: List[str] = field(default_factory=list)

    @classmethod
    def from_result_and_cfg(
        cls,
        result: PCSelectionResult,
        rs_cfg: dict,
        ep_cfg: dict,
    ) -> "SignalConfig":
        return cls(
            entry_threshold       = result.entry_threshold,
            signal_direction      = result.signal_direction,
            exit_threshold_signal = float(rs_cfg["exit_threshold_signal"]),
            max_hold_days         = int(rs_cfg.get("max_hold_days", ep_cfg["halflife_primary"])),
            strategies = list(rs_cfg.get("strategies", ["A", "B", "C"])),
        )


# ---------------------------------------------------------------------------
# Exit condition helper
# ---------------------------------------------------------------------------

def _signal_exit_triggered(
    dar_t:     float,
    threshold: float,
    direction: int,
) -> bool:
    """
    True when the Strategy B exit condition is met at time t.

    For direction=+1: entered when delta_ar was LOW (< entry_threshold).
                      Exit when it recovers HIGH (> exit_threshold_signal).
    For direction=-1: entered when delta_ar was HIGH (> entry_threshold).
                      Exit when it recovers LOW  (< exit_threshold_signal).
    """
    if direction == +1:
        return float(dar_t) > threshold
    return float(dar_t) < threshold


# ---------------------------------------------------------------------------
# Strategy state machines
# ---------------------------------------------------------------------------

def _generate_signal_a(
    in_regime:     np.ndarray,
    max_hold_days: int,
) -> np.ndarray:
    """
    Strategy A: hard time-based exit.

    Enter on the first bar where in_regime=True (while not already in
    position). Hold for exactly max_hold_days bars, then go flat.
    Re-enter on the next bar if in_regime is still True.

    Parameters
    ----------
    in_regime     : (T,) boolean array — True on days entry condition holds.
    max_hold_days : number of bars to hold before forcing exit.

    Returns
    -------
    signal : (T,) float array — 1.0 when long, 0.0 when flat.
    """
    T           = len(in_regime)
    signal      = np.zeros(T, dtype=float)
    in_position = False
    days_held   = 0

    for t in range(T):
        if not in_position:
            if in_regime[t]:
                in_position = True
                days_held   = 1
                signal[t]   = 1.0
        else:
            signal[t] = 1.0
            if days_held >= max_hold_days:
                in_position = False
                days_held   = 0
            else:
                days_held += 1

    return signal


def _generate_signal_b(
    in_regime:  np.ndarray,
    delta_ar:   np.ndarray,
    exit_threshold: float,
    direction:  int,
) -> np.ndarray:
    """
    Strategy B: trailing signal-based exit.

    Enter when in_regime=True. Hold until the signal exit condition fires
    (delta_ar recovers past exit_threshold_signal). For direction=+1 with
    entry=-1.0σ and exit=-0.5σ, a position is never closed on its entry bar
    (hysteresis). Handles direction=-1 symmetrically.

    Parameters
    ----------
    in_regime      : (T,) boolean array from regime_classifier.
    delta_ar       : (T,) standardised ΔAR signal.
    exit_threshold : z-score level for exit (from rs_cfg["exit_threshold_signal"]).
    direction      : +1 or -1 from PCSelectionResult.

    Returns
    -------
    signal : (T,) float array — 1.0 when long, 0.0 when flat.
    """
    T           = len(in_regime)
    signal      = np.zeros(T, dtype=float)
    in_position = False

    for t in range(T):
        if not in_position:
            if in_regime[t]:
                in_position = True
                signal[t]   = 1.0
                # Generic same-bar exit guard — matters for direction=-1
                # where exit threshold could theoretically be met on entry bar.
                if _signal_exit_triggered(delta_ar[t], exit_threshold, direction):
                    in_position = False
        else:
            signal[t] = 1.0
            if _signal_exit_triggered(delta_ar[t], exit_threshold, direction):
                in_position = False

    return signal


def _generate_signal_c(
    in_regime:      np.ndarray,
    delta_ar:       np.ndarray,
    exit_threshold: float,
    direction:      int,
    max_hold_days:  int,
) -> np.ndarray:
    """
    Strategy C: combination exit; whichever of A or B fires first.

    Entry is identical to A and B. Exit fires when EITHER:
      - days_held >= max_hold_days  (Strategy A rule), OR
      - signal exit condition met   (Strategy B rule).

    Parameters
    ----------
    in_regime      : (T,) boolean array from regime_classifier.
    delta_ar       : (T,) standardised ΔAR signal.
    exit_threshold : z-score level for signal exit.
    direction      : +1 or -1 from PCSelectionResult.
    max_hold_days  : hard exit after N bars.

    Returns
    -------
    signal : (T,) float array — 1.0 when long, 0.0 when flat.
    """
    T           = len(in_regime)
    signal      = np.zeros(T, dtype=float)
    in_position = False
    days_held   = 0

    for t in range(T):
        if not in_position:
            if in_regime[t]:
                in_position = True
                days_held   = 1
                signal[t]   = 1.0
                exit_now = (
                    _signal_exit_triggered(delta_ar[t], exit_threshold, direction)
                    or days_held >= max_hold_days
                )
                if exit_now:
                    in_position = False
                    days_held   = 0
        else:
            signal[t] = 1.0
            exit_now = (
                _signal_exit_triggered(delta_ar[t], exit_threshold, direction)
                or days_held >= max_hold_days
            )
            if exit_now:
                in_position = False
                days_held   = 0
            else:
                days_held += 1

    return signal


# ---------------------------------------------------------------------------
# Trade statistics
# ---------------------------------------------------------------------------

def _compute_trade_stats(signal: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for a single strategy's signal array.

    n_trades       : number of distinct entry events (out → in transitions).
    avg_hold_days  : mean holding period in bars per trade.
    pct_in_market  : percentage of total bars spent long.

    These are used to detect degenerate strategies:
      - n_trades = 0  → strategy never fires, threshold too tight
      - pct_in_market > 80  → strategy is almost always in, adds no filtering value
    """
    in_pos    = signal > 0.5
    entries   = np.diff(in_pos.astype(int), prepend=0)
    n_trades  = int((entries == 1).sum())

    pct_in = float(in_pos.mean() * 100)
    avg_hold = float(in_pos.sum()) / n_trades if n_trades > 0 else 0.0

    return {
        "n_trades":       n_trades,
        "avg_hold_days":  round(avg_hold, 1),
        "pct_in_market":  round(pct_in, 1),
    }


# ---------------------------------------------------------------------------
# Core signal generation
# ---------------------------------------------------------------------------

def generate_signals(
    labelled_panel: pl.DataFrame,
    result:         PCSelectionResult,
    rs_cfg:         dict,
    ep_cfg:         dict,
) -> pl.DataFrame:
    """
    Apply all three exit strategies to the labelled panel.

    Parameters
    ----------
    labelled_panel : output of regime_classifier.label_regimes().
                     Must contain columns: in_regime (bool), delta_ar (f64).
    result         : PCSelectionResult from regime_classifier.select_best_pc().
    rs_cfg         : cfg["regime_strats"] block.

    Returns
    -------
    signal_df : labelled_panel with signal_A, signal_B, signal_C columns
                appended (only for strategies listed in rs_cfg["strategies"]).

    Notes
    -----
    Signals are 0.0 (flat) or 1.0 (long). Short positions are never generated.
    The 1-day execution lag is applied by the backtester, not here.
    State machines run over the full panel without split boundary resets.
    """
    cfg        = SignalConfig.from_result_and_cfg(result, rs_cfg, ep_cfg)
    in_regime  = labelled_panel["in_regime"].to_numpy().astype(bool)
    delta_ar   = labelled_panel["delta_ar"].to_numpy().astype(float)
    strategies = cfg.strategies

    new_cols: Dict[str, List[float]] = {}

    if "A" in strategies:
        new_cols["signal_A"] = _generate_signal_a(
            in_regime, cfg.max_hold_days
        ).tolist()

    if "B" in strategies:
        new_cols["signal_B"] = _generate_signal_b(
            in_regime, delta_ar,
            cfg.exit_threshold_signal,
            cfg.signal_direction,
        ).tolist()

    if "C" in strategies:
        new_cols["signal_C"] = _generate_signal_c(
            in_regime, delta_ar,
            cfg.exit_threshold_signal,
            cfg.signal_direction,
            cfg.max_hold_days,
        ).tolist()

    return labelled_panel.with_columns([
        pl.Series(name, values)
        for name, values in new_cols.items()
    ])


# ---------------------------------------------------------------------------
# Diagnostic printer
# ---------------------------------------------------------------------------

def print_signal_summary(
    signal_df:   pl.DataFrame,
    split_masks: dict,
    strategies:  List[str],
) -> None:
    """
    Print per-strategy, per-split statistics to stdout.

    Flags degenerate strategies:
      - n_trades == 0       → WARN: threshold too tight, strategy never fires
      - pct_in_market > 80  → WARN: almost always in market, no filtering value
    """
    sep = "=" * 62
    print(f"\n{sep}")
    print("  SIGNAL GENERATOR SUMMARY")
    print(sep)

    split_labels = ["train", "val", "test", "full"]
    dates_arr    = signal_df["date"].to_numpy()

    # Build index arrays for each split — use date strings for alignment
    full_mask = np.ones(len(signal_df), dtype=bool)
    masks_ext = {**split_masks, "full": full_mask}

    for strat in strategies:
        col = f"signal_{strat}"
        if col not in signal_df.columns:
            continue

        sig = signal_df[col].to_numpy().astype(float)
        print(f"\n  Strategy {strat}"
              f"  (entry={signal_df['in_regime'].sum()} regime days in full panel)")

        header = f"    {'Split':<8}  {'Trades':>7}  {'Avg Hold':>9}  {'% In Mkt':>9}"
        print(header)
        print(f"    {'-'*8}  {'-'*7}  {'-'*9}  {'-'*9}")

        for label in split_labels:
            mask  = masks_ext[label]
            # Align mask length to panel (panel may be shorter than full df)
            if len(mask) != len(sig):
                print(f"    {label:<8}  {'mask length mismatch':>28}")
                continue

            stats = _compute_trade_stats(sig[mask])
            warn  = ""
            if stats["n_trades"] == 0:
                warn = "  ← WARN: never fires"
            elif stats["pct_in_market"] > 80:
                warn = "  ← WARN: > 80% in market"

            print(
                f"    {label:<8}  "
                f"{stats['n_trades']:>7}  "
                f"{stats['avg_hold_days']:>8.1f}d  "
                f"{stats['pct_in_market']:>8.1f}%"
                f"{warn}"
            )

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_signal_generator(
    labelled_panel: pl.DataFrame,
    split_masks:    dict,
    result:         PCSelectionResult,
    rs_cfg:         dict,
    ep_cfg:         dict,
) -> pl.DataFrame:
    """
    Full signal generation pipeline: apply exit rules → print stats → return.

    Parameters
    ----------
    labelled_panel : output of run_pc_selection() from regime_classifier.py.
    split_masks    : dict from get_split_masks(panel["date"].to_list(), cfg["splits"]).
                     Must be aligned to panel dates, not full df dates.
    result         : PCSelectionResult — fixed hyperparameters from train.
    rs_cfg         : cfg["regime_strats"] block from config.yaml.

    Returns
    -------
    signal_df : labelled_panel with signal_A, signal_B, signal_C appended.
                This is the primary input to the backtester.
    """
    strategies = rs_cfg.get("strategies", ["A", "B", "C"])
    print(f"  Generating signals for strategies: {strategies}")
    print(f"    Entry threshold   : {result.entry_threshold:+.1f}σ  "
          f"(direction={result.signal_direction:+d})")
    print(f"    Exit (B/C signal) : {rs_cfg['exit_threshold_signal']:+.1f}σ")
    print(f"    Exit (A/C time)   : {rs_cfg.get('max_hold_days', ep_cfg['halflife_primary'])} days")

    signal_df = generate_signals(labelled_panel, result, rs_cfg, ep_cfg)
    print_signal_summary(signal_df, split_masks, strategies)

    return signal_df
