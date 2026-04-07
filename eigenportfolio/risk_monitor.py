import numpy as np
import yaml
from typing import Dict, Tuple


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Absorption Ratio Anomaly Signal
# ---------------------------------------------------------------------------

def compute_delta_ar(
    absorption_ratios: np.ndarray,
    short_window: int,
    long_window: int,
) -> np.ndarray:
    """
    Standardised absorption ratio shift:

        delta_AR_t = (AR_short_t - AR_long_t) / std(AR_long_t)

    AR_short = short-window rolling mean of AR.
    AR_long / std = long-window rolling mean and standard deviation.

    A large positive value signals rapid correlation convergence (fragile /
    crisis-like regime). A large negative value signals rising diversification.
    NaN is returned for timesteps before the long_window is populated.
    """
    T = len(absorption_ratios)
    delta_ar = np.full(T, np.nan)

    for t in range(long_window, T):
        ar_short = float(np.mean(absorption_ratios[max(0, t - short_window): t]))
        long_slice = absorption_ratios[t - long_window: t]
        ar_long = float(np.mean(long_slice))
        ar_std = float(np.std(long_slice, ddof=1))
        if ar_std > 1e-12:
            delta_ar[t] = (ar_short - ar_long) / ar_std

    return delta_ar


# ---------------------------------------------------------------------------
# Option A: Variance Explained
# ---------------------------------------------------------------------------

def compute_ew_baseline(sigma: np.ndarray) -> float:
    """
    Fraction of total cross-sectional variance explained by a single
    equal-weighted factor, derived analytically from the covariance matrix.

    From a single-factor OLS model r_i = beta_i * F + epsilon_i,
    where F = w^T r and w = (1/N) * ones(N):

        beta_i   = (Sigma @ w)_i / (w^T Sigma w)
        Fraction = sum_i(beta_i^2 * Var(F)) / trace(Sigma)
                 = ||Sigma @ w||^2 / (w^T Sigma w * trace(Sigma))

    Computed purely from the rolling Sigma — no return data needed.
    """
    N = sigma.shape[0]
    w = np.ones(N) / N
    sigma_w = sigma @ w
    numerator = float(sigma_w @ sigma_w)                # ||Sigma w||^2
    denominator = float(w @ sigma_w) * np.trace(sigma)  # (w^T Sigma w) * tr(Sigma)
    if denominator < 1e-12:
        return 0.0
    return numerator / denominator


def compute_champion_vs_challenger(
    cov_series: np.ndarray,
    absorption_ratios: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute the Champion vs Challenger comparison at every timestep.

    Challenger (PCA)       : absorption_ratio — fraction of total variance in top-K PCs.
    Baseline (Equal-Weight): fraction of total variance explained by single EW factor.

    Returns
    -------
    dict with:
        pca_explained : (T,) absorption ratios (Challenger)
        ew_explained  : (T,) equal-weight factor explained fraction (Baseline)
        pca_edge      : (T,) pca_explained - ew_explained (positive = PCA wins)
    """
    T = cov_series.shape[0]
    ew_explained = np.array([compute_ew_baseline(cov_series[t]) for t in range(T)])
    return {
        "pca_explained": absorption_ratios,
        "ew_explained": ew_explained,
        "pca_edge": absorption_ratios - ew_explained,
    }


# ---------------------------------------------------------------------------
# Top-level monitor
# ---------------------------------------------------------------------------

def run_risk_monitor(
    decomp_results: Dict[int, Dict],
    cov_results: Dict[int, Tuple[np.ndarray, int]],
    short_window: int,
    long_window: int,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Run the full risk monitor across all half-life configurations.

    Returns
    -------
    dict mapping half_life -> {delta_ar, pca_explained, ew_explained, pca_edge}
    """
    results: Dict[int, Dict[str, np.ndarray]] = {}
    for hl, decomp in decomp_results.items():
        cov_series, _ = cov_results[hl]
        ar = decomp["absorption_ratios"]
        cvc = compute_champion_vs_challenger(cov_series, ar)
        results[hl] = {
            "delta_ar": compute_delta_ar(ar, short_window, long_window),
            **cvc,
        }
    return results
