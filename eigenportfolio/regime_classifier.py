"""
regime_classifier.py  —  PC Selection and Regime Labelling
===========================================================
Component 0 of the regime_strats pipeline (Phase 4).

Responsibilities
----------------
1. PC Selection  : grid-search over (PC index k, direction, entry threshold)
                   on TRAIN data only. Selects the combination that maximises
                   conditional Sharpe ratio subject to a minimum observation
                   count guard.

2. Regime Labels : applies the selected parameters to the FULL panel, adding
                   `in_regime` (bool) and `signal` (float) columns. The
                   backtester shifts signal by 1 day before constructing
                   positions; no look-ahead is introduced here.

Key design constraint
---------------------
Every decision is derived from training data via the grid search. 
For the current 9-sector ETF universe the selection will recover 
PC2 / direction=+1 / threshold=-1.0σ, which is consistent with the
diagnostic results. For a different universe, it
will find whatever is actually there.
"""

import numpy as np
import polars as pl
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class PCSelectionResult:
    """
    Immutable record of the PC selection outcome on train data.

    These fields become the fixed hyperparameters for all of Phase 4.
    They must never be re-estimated on val or test data.

    Attributes
    ----------
    best_pc          : 0-indexed PC column that maximises conditional SR on train.
                       Translates to panel column f"pc{best_pc + 1}_ret".
    signal_direction : +1 → long when ΔAR < entry_threshold (diversifying regime).
                       -1 → long when ΔAR > entry_threshold (fragile / crisis regime).
    entry_threshold  : ΔAR z-score level that triggers entry.
    cond_sr          : annualised conditional SR of best combo on train.
    uncond_sr        : annualised unconditional SR of best PC on train (baseline).
    sr_improvement   : cond_sr − uncond_sr (the value added by conditioning).
    n_regime_days    : number of train days spent in the active regime.
    selection_table  : full grid results as a Polars DataFrame for inspection.
    """
    best_pc:          int
    signal_direction: int
    entry_threshold:  float
    cond_sr:          float
    uncond_sr:        float
    sr_improvement:   float
    n_regime_days:    int
    selection_table:  pl.DataFrame = field(repr=False)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_conditional_sr(
    pc_returns: np.ndarray,
    delta_ar:   np.ndarray,
    threshold:  float,
    direction:  int,
    min_obs:    int = 20,
) -> Tuple[float, int]:
    """
    Annualised Sharpe ratio of pc_returns on days where the regime is active.

    The regime is active when:
      direction = +1 : delta_ar < threshold   (diversifying regime)
      direction = -1 : delta_ar > threshold   (fragile / crisis regime)

    Parameters
    ----------
    pc_returns : (T,) daily eigenportfolio returns for one PC
    delta_ar   : (T,) standardised AR shift signal (z-score units)
    threshold  : entry level in z-score units
    direction  : +1 or -1 (see above)
    min_obs    : minimum regime days required; returns (nan, count) if not met

    Returns
    -------
    (cond_sr, n_regime_days)
        cond_sr      : annualised Sharpe on regime days, or nan if insufficient data
        n_regime_days: number of days the regime was active
    """
    mask = (delta_ar < threshold) if direction == +1 else (delta_ar > threshold)
    regime_rets = pc_returns[mask]
    n = int(mask.sum())

    if n < min_obs:
        return (np.nan, n)

    mu  = float(np.mean(regime_rets))
    sig = float(np.std(regime_rets, ddof=1))

    if sig < 1e-12:
        return (np.nan, n)

    return (mu / sig * np.sqrt(252), n)


def _compute_unconditional_sr(pc_returns: np.ndarray) -> float:
    """Annualised Sharpe ratio with no conditioning (full-period baseline)."""
    mu  = float(np.mean(pc_returns))
    sig = float(np.std(pc_returns, ddof=1))
    if sig < 1e-12:
        return np.nan
    return mu / sig * np.sqrt(252)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def select_best_pc(
    panel_train: pl.DataFrame,
    rs_cfg:      dict,
) -> PCSelectionResult:
    """
    Grid-search over (PC index, direction, threshold) on TRAIN data only.

    Iterates over every PC column in panel_train, both signal directions, and
    every candidate threshold from rs_cfg. Selects the combination with the
    highest conditional SR subject to the min_observations guard.

    Parameters
    ----------
    panel_train : Polars DataFrame — train split of the aligned panel.
                  Must contain columns pc1_ret ... pcK_ret and delta_ar.
    rs_cfg      : cfg["regime_strats"] block from config.yaml

    Returns
    -------
    PCSelectionResult — all fields set from train data, ready for Phase 4.

    Raises
    ------
    RuntimeError if no combination passes the min_observations threshold.
    """
    thresholds = rs_cfg.get("candidate_thresholds", [-2.0, -1.5, -1.0, -0.5])
    min_obs    = rs_cfg.get("min_observations", 20)

    pc_cols  = sorted([c for c in panel_train.columns if c.endswith("_ret")])
    delta_ar = panel_train["delta_ar"].to_numpy()

    rows: List[dict] = []

    for k_idx, col in enumerate(pc_cols):
        pc_rets   = panel_train[col].to_numpy()
        uncond_sr = _compute_unconditional_sr(pc_rets)

        for direction in [+1, -1]:
            for threshold in thresholds:
                cond_sr, n = compute_conditional_sr(
                    pc_rets, delta_ar, threshold, direction, min_obs
                )
                sr_imp = (cond_sr - uncond_sr) if not np.isnan(cond_sr) else np.nan

                rows.append({
                    "pc_col":         col,
                    "k_idx":          k_idx,
                    "direction":      direction,
                    "threshold":      float(threshold),
                    "cond_sr":        round(float(cond_sr), 4) if not np.isnan(cond_sr) else None,
                    "uncond_sr":      round(float(uncond_sr), 4),
                    "sr_improvement": round(float(sr_imp), 4) if not np.isnan(sr_imp) else None,
                    "n_regime_days":  n,
                })

    table = pl.DataFrame(rows)

    valid = table.filter(pl.col("cond_sr").is_not_null())
    if len(valid) == 0:
        raise RuntimeError(
            "PC selection failed: no (k, direction, threshold) combination "
            f"produced >= {min_obs} regime days on the training set. "
            "Lower min_observations in config.yaml or provide more training data."
        )

    best = valid.sort("cond_sr", descending=True).row(0, named=True)

    return PCSelectionResult(
        best_pc          = best["k_idx"],
        signal_direction = best["direction"],
        entry_threshold  = best["threshold"],
        cond_sr          = best["cond_sr"],
        uncond_sr        = best["uncond_sr"],
        sr_improvement   = best["sr_improvement"],
        n_regime_days    = best["n_regime_days"],
        selection_table  = table,
    )


# ---------------------------------------------------------------------------
# Regime labelling
# ---------------------------------------------------------------------------

def label_regimes(
    panel:           pl.DataFrame,
    result:          PCSelectionResult,
    exit_threshold:  Optional[float] = None,
) -> pl.DataFrame:
    """
    Apply selected PC parameters to the full panel, appending regime columns.

    Parameters
    ----------
    panel          : full aligned panel (all splits) from build_aligned_panel().
    result         : PCSelectionResult from select_best_pc().
    exit_threshold : trailing exit level from rs_cfg["exit_threshold_signal"].
                     Currently stored as metadata only — used by the backtester.
                     If None, signal is a simple binary in/out flag.

    Returns
    -------
    panel with three additional columns:
        selected_pc_ret : float — returns of the selected PC (convenience alias)
        in_regime       : bool  — True on days the entry condition is satisfied
        signal          : float — +1.0 (in regime) or 0.0 (out of regime)

    Notes on look-ahead
    -------------------
    in_regime[t] is determined solely by delta_ar[t], which is computed from
    covariance data up to and including day t. No future information enters.
    The backtester applies a 1-day execution lag by shifting signal forward
    before constructing positions.
    """
    pc_col    = f"pc{result.best_pc + 1}_ret"
    dar       = panel["delta_ar"].to_numpy()
    direction = result.signal_direction
    entry     = result.entry_threshold

    if direction == +1:
        in_regime = dar < entry
    else:
        in_regime = dar > entry

    return panel.with_columns([
        pl.Series("selected_pc_ret", panel[pc_col].to_list()),
        pl.Series("in_regime",       in_regime.tolist()),
        pl.Series("signal",          in_regime.astype(float).tolist()),
    ])


# ---------------------------------------------------------------------------
# Diagnostic printer
# ---------------------------------------------------------------------------

def print_selection_summary(result: PCSelectionResult) -> None:
    """Print a formatted summary of PC selection results to stdout."""
    pc_label = f"PC{result.best_pc + 1}"
    dir_desc = (
        "long when ΔAR < threshold  (diversifying regime)"
        if result.signal_direction == +1
        else "long when ΔAR > threshold  (fragile / crisis regime)"
    )

    sep = "=" * 62
    print(f"\n{sep}")
    print("  PC SELECTION RESULTS  (train data only — never re-estimated)")
    print(sep)
    print(f"  Selected PC      :  {pc_label}  (0-indexed: {result.best_pc})")
    print(f"  Direction        :  {result.signal_direction:+d}   {dir_desc}")
    print(f"  Entry threshold  :  {result.entry_threshold:+.1f}σ")
    print(f"  Conditional SR   :  {result.cond_sr:+.3f}")
    print(f"  Unconditional SR :  {result.uncond_sr:+.3f}")
    print(f"  SR improvement   :  {result.sr_improvement:+.3f}  (conditioning value-add)")
    print(f"  Regime days      :  {result.n_regime_days}")
    print(sep)

    valid = result.selection_table.filter(pl.col("cond_sr").is_not_null())
    top10 = valid.sort("cond_sr", descending=True).head(10)
    print("\n  Top 10 candidates (all splits, sorted by conditional SR on train):")
    print(
        top10.select([
            "pc_col", "direction", "threshold",
            "cond_sr", "uncond_sr", "sr_improvement", "n_regime_days",
        ]).to_pandas().to_string(index=False)
    )
    print()


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_pc_selection(
    panel:       pl.DataFrame,
    split_masks: dict,
    rs_cfg:      dict,
) -> Tuple[PCSelectionResult, pl.DataFrame]:
    """
    Full PC selection pipeline: train-only selection → full-panel labelling.

    Parameters
    ----------
    panel        : full aligned panel from build_aligned_panel().
                   split_masks must be aligned to THIS panel's dates,
                   not to the full df dates from preprocess().
    split_masks  : dict from get_split_masks(panel["date"].to_list(), cfg["splits"]).
                   Keys: "train", "val", "test" — each a boolean np.ndarray
                   with length == len(panel).
    rs_cfg       : cfg["regime_strats"] block from config.yaml.

    Returns
    -------
    result         : PCSelectionResult — fixed hyperparameters for Phase 4.
    labelled_panel : full panel with in_regime and signal columns appended.
    """
    train_mask  = split_masks["train"]
    panel_train = panel.filter(pl.Series("_mask", train_mask))

    print(f"  PC selection on train split: {len(panel_train)} days")
    print(f"  Val days: {int(split_masks['val'].sum())}  |  "
          f"Test days: {int(split_masks['test'].sum())}")

    result = select_best_pc(panel_train, rs_cfg)
    print_selection_summary(result)

    exit_threshold = rs_cfg.get("exit_threshold_signal", None)
    labelled_panel = label_regimes(panel, result, exit_threshold)

    return result, labelled_panel
