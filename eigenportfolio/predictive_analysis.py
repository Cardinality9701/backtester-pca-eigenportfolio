"""
predictive_analysis.py - Delta-AR vs Forward VIX Predictive Regression
=======================================================================
Tests whether the delta absorption ratio (ΔAR) has leading predictive
power over the VIX index at forward horizons h ∈ {1, 5, 21} trading days.

Research hypothesis
-------------------
ΔAR is a *concurrent* indicator of correlation regime, not a leading
predictor of volatility.  It justifies regime-labelling but should not
be used for directional VIX forecasting.

Expected results:
  - Full-sample R² is low across all horizons (< 0.05 is typical)
  - R² decays as h increases: R²(h=1) > R²(h=5) > R²(h=21)
  - Rolling R² spikes during stress events (2008, 2020, 2022) where the
    ΔAR–VIX relationship is temporarily stronger
  - p-value may appear significant at short horizons due to autocorrelation
    in both series, but economic magnitude remains small

Inputs (from main.py Step 7)
-----------------------------
  delta_ar   : (T_eff,) ΔAR from monitor_results[vix_hl]["delta_ar"]
               Contains leading NaNs (first long_window=252 positions).
  vix_series : (T_eff,) VIX levels, pre-aligned to delta_ar. No NaNs.
  dates      : list of T_eff dates, pre-aligned to both arrays.
  cfg        : vix block from config.yaml.

Return contract
---------------
{
  "dates":         List[date]         aligned dates after NaN rows are dropped
  "delta_ar":      np.ndarray (T_v,)  concurrent ΔAR values
  "vix_levels":    np.ndarray (T_v,)  concurrent VIX levels
  "vix_regime":    np.ndarray (T_v,)  0=Low / 1=Elevated / 2=Stress  (int)
  "regime_labels": List[str]          e.g. ["Low (<20)", "Elevated (20-30)", "Stress (>30)"]
  "thresholds":    List[float]        e.g. [20.0, 30.0]
  "ols":           {h: OLSResult}     full-sample regression result per horizon h
  "rolling_r2":    {h: (dates, r2)}   rolling-window R² series per horizon h
}

OLSResult keys: slope, intercept, r2, p_value, n_obs, x_fit, y_fit
rolling_r2   : {h: (List[date], np.ndarray)} where both have length n_pairs - window
"""

import numpy as np
from scipy import stats
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _drop_nan_aligned(
    delta_ar:   np.ndarray,
    vix_series: np.ndarray,
    dates:      list,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Remove positions where delta_ar is NaN, applying the same mask to all
    three arrays so they remain date-aligned.

    delta_ar has leading NaNs (first long_window=252 values) because
    compute_delta_ar() in risk_monitor.py requires a full long-window before
    it can compute the first standardised value.  vix_series has no NaNs —
    this is guaranteed by data_preprocessor.py's forward-fill and drop_nulls.
    """
    assert len(delta_ar) == len(vix_series) == len(dates), (
        f"Length mismatch: delta_ar={len(delta_ar)}, "
        f"vix_series={len(vix_series)}, dates={len(dates)}"
    )
    valid = ~np.isnan(delta_ar)
    return (
        delta_ar[valid],
        vix_series[valid],
        [d for d, v in zip(dates, valid) if v],
    )


def _assign_vix_regime(
    vix_levels: np.ndarray,
    thresholds: List[float],
) -> np.ndarray:
    """
    Assign integer regime label to each VIX observation.

      0 = Low       (VIX < thresholds[0])
      1 = Elevated  (thresholds[0] <= VIX < thresholds[1])
      2 = Stress    (VIX >= thresholds[1])

    Assignment is done from lowest to highest threshold so that the final
    label is always the highest applicable bucket.
    """
    regime = np.zeros(len(vix_levels), dtype=int)
    regime[vix_levels >= thresholds[0]] = 1
    regime[vix_levels >= thresholds[1]] = 2
    return regime


def _regime_label_strings(thresholds: List[float]) -> List[str]:
    lo, hi = float(thresholds[0]), float(thresholds[1])
    return [f"Low (<{lo:.0f})", f"Elevated ({lo:.0f}–{hi:.0f})", f"Stress (>{hi:.0f})"]


# ---------------------------------------------------------------------------
# Full-sample OLS:  VIX_{t+h} ~ ΔAR_t
# ---------------------------------------------------------------------------

def run_full_sample_ols(
    delta_ar_valid: np.ndarray,
    vix_valid:      np.ndarray,
    forward_horizons: List[int],
) -> Dict[int, Dict[str, Any]]:
    """
    Full-sample OLS for each forward horizon h.

    Specification: VIX_{t+h} = α + β · ΔAR_t + ε_t

    Alignment
    ---------
      X = delta_ar_valid[:-h]   (ΔAR at time t,   length T_v - h)
      Y = vix_valid[h:]         (VIX at time t+h,  length T_v - h)

    Both arrays are already NaN-free (guaranteed by _drop_nan_aligned).
    All (X, Y) pairs are strictly historical at evaluation time — zero
    lookahead for a descriptive regression.

    Also computes x_fit / y_fit (200 evenly-spaced points covering the
    range of X) so visualizer.py can overlay the OLS line on the scatter
    plot without re-fitting.

    Returns
    -------
    {h: {"slope", "intercept", "r2", "p_value", "n_obs", "x_fit", "y_fit"}}
    """
    T_v     = len(delta_ar_valid)
    results: Dict[int, Dict[str, Any]] = {}

    for h in forward_horizons:
        if h >= T_v:
            continue
        X = delta_ar_valid[:-h]
        Y = vix_valid[h:]

        slope, intercept, r, p_value, _ = stats.linregress(X, Y)
        r2 = float(r ** 2)

        # OLS line for scatter overlay — range of observed X
        x_fit = np.linspace(float(X.min()), float(X.max()), 200)
        y_fit = intercept + slope * x_fit

        results[h] = {
            "slope":     float(slope),
            "intercept": float(intercept),
            "r2":        r2,
            "p_value":   float(p_value),
            "n_obs":     int(len(X)),
            "x_fit":     x_fit,
            "y_fit":     y_fit,
        }

    return results


# ---------------------------------------------------------------------------
# Rolling OLS R²
# ---------------------------------------------------------------------------

def run_rolling_ols(
    delta_ar_valid:   np.ndarray,
    vix_valid:        np.ndarray,
    dates_valid:      list,
    forward_horizons: List[int],
    rolling_window:   int = 252,
    min_obs:          int = 30,
) -> Dict[int, Tuple[list, np.ndarray]]:
    """
    Rolling window OLS R² series for each forward horizon h.

    At each step i (i >= rolling_window), OLS is fit on the past
    rolling_window (X, Y) pairs:
      X_window = delta_ar_valid[i - window : i]
      Y_window = vix_valid     [i - window : i]   (= vix at t+h for each t)

    The (X, Y) pairing is constructed once before the loop:
      X_full = delta_ar_valid[:n_pairs]   where n_pairs = T_v - h
      Y_full = vix_valid[h:T_v]

    This ensures Y_full[j] = VIX h days after X_full[j] for every j,
    so the window [i-window:i] always contains matched (ΔAR_t, VIX_{t+h})
    pairs with no date drift.

    Parameters 
    ----------
    rolling_window : OLS lookback in trading days (from config.yaml)
    min_obs        : minimum non-NaN X values needed to fit OLS;
                     NaN is returned for steps with fewer observations

    Returns
    -------
    {h: (dates_of_x_obs, r2_array)}
      dates_of_x_obs : List of length n_pairs - rolling_window
      r2_array       : np.ndarray of same length (may contain NaN)
    """
    T_v  = len(delta_ar_valid)
    out: Dict[int, Tuple[list, np.ndarray]] = {}

    for h in forward_horizons:
        n_pairs = T_v - h
        if n_pairs <= rolling_window:
            continue

        # Pre-build aligned (X, Y) pairs across the full valid series
        X_full   = delta_ar_valid[:n_pairs]   # X = ΔAR_t
        Y_full   = vix_valid[h:T_v]           # Y = VIX_{t+h}; same length as X_full
        dates_x  = dates_valid[:n_pairs]      # date index for X obs

        r2_arr        = np.full(n_pairs, np.nan)
        result_dates: list = []

        for i in range(rolling_window, n_pairs):
            X_w = X_full[i - rolling_window : i]
            Y_w = Y_full[i - rolling_window : i]

            # Defensive NaN filter within the window.
            # After _drop_nan_aligned this should never trigger, but guards
            # against any edge-case NaNs introduced by future code changes.
            valid = ~np.isnan(X_w)
            result_dates.append(dates_x[i])
            if valid.sum() < min_obs:
                continue                      # r2_arr[i] stays NaN

            _, _, r, _, _ = stats.linregress(X_w[valid], Y_w[valid])
            r2_arr[i] = float(r ** 2)

        # Slice away the leading NaN warmup so the returned array starts at
        # the first computed value and aligns with result_dates.
        r2_slice = r2_arr[rolling_window:]    # length = n_pairs - rolling_window
        out[h]   = (result_dates, r2_slice)

    return out


# ---------------------------------------------------------------------------
# Stdout summary printer
# ---------------------------------------------------------------------------

def _print_summary(
    ols_results: Dict[int, Dict],
    thresholds:  List[float],
    vix_regime:  np.ndarray,
) -> None:
    labels = _regime_label_strings(thresholds)
    print("\n   Predictive regression summary  (VIX_{t+h} ~ ΔAR_t):")
    print(f"   {'h':>5}  {'β (slope)':>10}  {'R²':>7}  {'p-value':>10}  {'n_obs':>7}")
    print("   " + "-" * 55)
    for h in sorted(ols_results.keys()):
        r   = ols_results[h]
        sig = "**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else "  ")
        print(f"   {h:>3}d{sig}  {r['slope']:>+10.4f}  {r['r2']:>7.4f}"
              f"  {r['p_value']:>10.4f}  {r['n_obs']:>7}")
    print("   ** p<0.01  * p<0.05")
    print("   Caution: autocorrelation in both series inflates significance.")
    print(f"\n   VIX regime distribution:")
    for i, lbl in enumerate(labels):
        count = int((vix_regime == i).sum())
        pct   = 100.0 * count / len(vix_regime)
        print(f"     {lbl:25s}: {count:5d} days  ({pct:5.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_predictive_analysis(
    delta_ar:   np.ndarray,
    vix_series: np.ndarray,
    dates:      list,
    cfg:        dict,
) -> dict:
    """
    Run the full Phase 3 predictive analysis pipeline.

    Parameters
    ----------
    delta_ar   : (T_eff,) ΔAR from monitor_results[vix_hl]["delta_ar"].
                 Contains leading NaNs (first long_window=252 positions).
    vix_series : (T_eff,) VIX levels aligned to delta_ar.  No NaNs.
    dates      : List of T_eff dates aligned to both arrays.
    cfg        : vix block from config.yaml.

    Returns
    -------
    See module docstring for the complete return contract.
    """
    forward_horizons = list(cfg.get("forward_horizons", [1, 5, 21]))
    thresholds       = list(cfg.get("level_thresholds", [20.0, 30.0]))
    rolling_window   = int(cfg.get("rolling_window", 252))

    # ── 1. Drop NaN rows ─────────────────────────────────────────────────────
    da_valid, vix_valid, dates_valid = _drop_nan_aligned(delta_ar, vix_series, dates)
    print(f"   NaN drop : {len(delta_ar) - len(da_valid)} leading rows removed "
          f"({len(da_valid)} observations remaining)")

    # ── 2. VIX regime labels ──────────────────────────────────────────────────
    vix_regime = _assign_vix_regime(vix_valid, thresholds)

    # ── 3. Full-sample OLS ────────────────────────────────────────────────────
    ols_results = run_full_sample_ols(da_valid, vix_valid, forward_horizons)

    # ── 4. Rolling OLS R² ─────────────────────────────────────────────────────
    rolling_r2 = run_rolling_ols(
        da_valid, vix_valid, dates_valid, forward_horizons, rolling_window
    )

    _print_summary(ols_results, thresholds, vix_regime)

    return {
        "dates":         dates_valid,
        "delta_ar":      da_valid,
        "vix_levels":    vix_valid,
        "vix_regime":    vix_regime,
        "regime_labels": _regime_label_strings(thresholds),
        "thresholds":    thresholds,
        "ols":           ols_results,
        "rolling_r2":    rolling_r2,
    }
