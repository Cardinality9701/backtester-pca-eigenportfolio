"""
covariance_engine.py — Rolling Covariance Estimators
=====================================================
Provides two families of covariance estimators used throughout the pipeline:

  1. EWM (Exponentially Weighted Moving) — recursive RiskMetrics update.
     Fast O(N²) per timestep, highly responsive to recent returns.
     Parametrised by half-life; four variants run in parallel.

  2. Ledoit-Wolf shrinkage — analytical shrinkage toward scaled identity.
     Uses a fixed rolling window; slower to adapt but more stable in
     calm regimes. Used as one leg of covariance comparison.

API convention (both families)
-------------------------------
  All estimators return Tuple[np.ndarray, int]:
      cov_series   : (T_eff, N, N)  rolling covariance matrices
      warmup_steps : int            leading rows of returns_matrix consumed

  All estimators treated interchangeably.
"""

import numpy as np
import yaml
from typing import Dict, List, Tuple

from sklearn.covariance import LedoitWolf


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# EWM covariance
# ---------------------------------------------------------------------------

def _lambda_from_halflife(half_life: int) -> float:
    """Decay factor λ = 2^(−1/H) so that weight halves every H days."""
    return 2.0 ** (-1.0 / half_life)


def compute_ewm_covariance_series(
    returns: np.ndarray,
    half_life: int,
    warmup_multiplier: int = 2,
) -> Tuple[np.ndarray, int]:
    """
    Rolling EWM covariance via the recursive RiskMetrics (zero-mean) update:

        Σ_t = λ Σ_{t−1} + (1 − λ) r_t r_t^T

    The first (warmup_multiplier × half_life) observations seed Σ via the
    sample covariance and are excluded from the returned series.

    Parameters
    ----------
    returns           : (T, N) array of log returns
    half_life         : EWM half-life in trading days
    warmup_multiplier : number of half-life periods consumed during warm-up

    Returns
    -------
    cov_series   : (T − warmup_steps, N, N) EWM covariance matrices
    warmup_steps : number of leading rows consumed by warm-up
    """
    T, N = returns.shape
    lam          = _lambda_from_halflife(half_life)
    warmup_steps = warmup_multiplier * half_life

    if warmup_steps >= T:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) >= T ({T}). "
            "Reduce warmup_multiplier or provide more data."
        )

    sigma = np.cov(returns[:warmup_steps].T)
    cov_list: List[np.ndarray] = []

    for t in range(warmup_steps, T):
        r     = returns[t, :, np.newaxis]              # (N, 1)
        sigma = lam * sigma + (1.0 - lam) * (r @ r.T)
        cov_list.append(sigma.copy())

    return np.array(cov_list), warmup_steps


def compute_all_covariance_series(
    returns: np.ndarray,
    half_lives: List[int],
    warmup_multiplier: int = 2,
) -> Dict[int, Tuple[np.ndarray, int]]:
    """
    Compute EWM covariance series for every half-life in the list.

    Returns
    -------
    dict mapping half_life -> (cov_series, warmup_steps)
    """
    return {
        hl: compute_ewm_covariance_series(returns, hl, warmup_multiplier)
        for hl in half_lives
    }

# ---------------------------------------------------------------------------
# Ledoit-Wolf covariance + shrinkage
# ---------------------------------------------------------------------------

def compute_ledoit_wolf_full(
    returns: np.ndarray,
    window: int = 252,
    assume_centered: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Rolling Ledoit-Wolf analytical shrinkage covariance series.

    Single pass; fits sklearn LedoitWolf on returns[t−window : t] at each
    timestep t, zero lookahead guaranteed.

    Shrinkage model:  Σ_LW = (1 − ρ) S + ρ μ I
        ρ  = analytically optimal shrinkage intensity ∈ [0, 1]
        S  = sample covariance of the window
        μ  = trace(S) / N

    Returns
    -------
    cov_series       : (T − window, N, N)  LW covariance matrices
    shrinkage_series : (T − window,)       shrinkage intensity ρ_t per step
    warmup_steps     : int = window
    """
    T, N = returns.shape
    warmup_steps = window

    if warmup_steps >= T:
        raise ValueError(
            f"window ({window}) >= T ({T}). "
            "Reduce window or provide more data."
        )

    lw               = LedoitWolf(assume_centered=assume_centered)
    T_eff            = T - warmup_steps
    cov_series       = np.empty((T_eff, N, N), dtype=np.float64)
    shrinkage_series = np.empty(T_eff,         dtype=np.float64)

    for t in range(warmup_steps, T):
        r_window                           = returns[t - warmup_steps : t, :]
        lw.fit(r_window)
        cov_series[t - warmup_steps]       = lw.covariance_
        shrinkage_series[t - warmup_steps] = lw.shrinkage_

    return cov_series, shrinkage_series, warmup_steps
