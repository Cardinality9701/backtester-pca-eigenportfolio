# =============================================================
# src/grids.py
# Centralised parameter grid definitions for all strategies.
#
# Used by:
#   - sensitivity.py  (grid search for heatmap)
#   - walk_forward.py (IS optimisation grid search)
#
# To add a new strategy:
#   1. Add an entry to STRATEGY_GRIDS
#   2. Set axis_a, axis_b, labels, param names, valid_rule
#   3. Set is_fast and is_slow default lists for walk-forward
#   No other files need to change.
# =============================================================
from __future__ import annotations

from typing import Any, Dict, List, Tuple


STRATEGY_GRIDS: Dict[str, Dict[str, Any]] = {

    "MovingAverageCross": {
        "axis_a":     [5, 10, 15, 20, 30],
        "axis_b":     [20, 40, 60, 80, 100, 120],
        "label_a":    "Fast Window (bars)",
        "label_b":    "Slow Window (bars)",
        "param_a":    "fast_window",
        "param_b":    "slow_window",
        "valid_rule": "a_lt_b",       # fast must be strictly < slow
        "is_fast":    [5, 10, 20, 30],
        "is_slow":    [40, 60, 80, 120],
    },

    "TSMOM": {
        "axis_a":     [126, 168, 210, 252, 315],
        "axis_b":     [5, 10, 21, 42, 63],
        "label_a":    "Lookback Window (bars)",
        "label_b":    "Skip Window (bars)",
        "param_a":    "lookback_window",
        "param_b":    "skip_window",
        "valid_rule": "b_lt_a",       # skip must be strictly < lookback
        "is_fast":    [126, 168, 210, 252, 315],
        "is_slow":    [5, 10, 21, 42],
    },

    "BollingerBand": {
        "axis_a":     [10, 15, 20, 25, 30],
        "axis_b":     [1.5, 2.0, 2.5, 3.0],
        "label_a":    "BB Window (bars)",
        "label_b":    "Num Std Devs",
        "param_a":    "window",
        "param_b":    "num_std",
        "valid_rule": "none",         # all (window, num_std) pairs are valid
        "is_fast":    [10, 15, 20, 25, 30],
        "is_slow":    [1.5, 2.0, 2.5],
    },

    "RSI": {
        "axis_a":     [7, 10, 14, 21, 28],
        "axis_b":     [20, 25, 30, 35],
        "label_a":    "RSI Period (bars)",
        "label_b":    "Oversold Threshold",
        "param_a":    "rsi_period",
        "param_b":    "oversold",
        # `overbought` is NOT swept — it stays fixed from base_params in config.
        # Sweeping oversold and overbought independently produces nonsensical
        # pairs (e.g. oversold=35, overbought=40). Fix overbought at 70.
        "valid_rule": "none",         # rsi_period and oversold are independent
        "is_fast":    [7, 10, 14, 21, 28],
        "is_slow":    [20, 25, 30, 35],
    },
}


def get_grid(strategy_type: str) -> Dict[str, Any]:
    """
    Return the full grid config dict for a given strategy type.
    Raises ValueError if the strategy has no registered grid.
    ValueError ignoreable for eigenportfolio strategies.
    """
    if strategy_type not in STRATEGY_GRIDS:
        raise ValueError(
            f"No grid defined for strategy '{strategy_type}'. "
            f"Available: {list(STRATEGY_GRIDS.keys())}"
        )
    return STRATEGY_GRIDS[strategy_type]


def get_is_windows(strategy_type: str) -> Tuple[List, List]:
    """
    Return (fast_windows, slow_windows) for walk-forward IS grid search.

    Naming convention by strategy:
      MovingAverageCross : fast = fast_window values, slow = slow_window values
      TSMOM              : fast = lookback values,    slow = skip values
      BollingerBand      : fast = window values,      slow = num_std values
      RSI                : fast = rsi_period values,  slow = oversold values
    """
    grid = get_grid(strategy_type)
    return grid["is_fast"], grid["is_slow"]


def is_valid_pair(strategy_type: str, a: float, b: float) -> bool:
    """
    Return True if (a, b) is a valid parameter combination for the strategy.

    Rules:
      a_lt_b : a must be strictly less than b  (MA Cross: fast < slow)
      b_lt_a : b must be strictly less than a  (TSMOM: skip < lookback)
      none   : all pairs valid                 (BollingerBand, RSI)
    """
    rule: str = STRATEGY_GRIDS[strategy_type]["valid_rule"]
    if rule == "a_lt_b":
        return a < b
    elif rule == "b_lt_a":
        return b < a
    elif rule == "none":
        return True
    else:
        raise ValueError(
            f"Unknown valid_rule '{rule}' for strategy '{strategy_type}'. "
            f"Expected one of: 'a_lt_b', 'b_lt_a', 'none'."
        )
