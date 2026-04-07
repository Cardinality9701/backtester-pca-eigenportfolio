"""
eigenportfolio.py — Eigenportfolio Construction and P&L Analysis
=================================================================
Constructs investable eigenportfolios from rolling PCA eigenvectors and
evaluates their out-of-sample P&L with strict zero-lookahead-bias.

Index alignment contract (critical)
------------------------------------
  decomp_results[hl]["eigenvectors"] has shape (T_cov, N, N) where
  T_cov = T - warmup_steps.  eigenvectors[t, :, k] is the k-th eigenvector
  of the EWM covariance matrix estimated from returns[0 : warmup + t + 1],
  i.e., using data up to and *including* returns_matrix[warmup + t].

  The first forward-looking eigenportfolio return is therefore:
      ep_return[0] = eigenvectors[0, :, k]^T @ returns_matrix[warmup + 1]

  This gives a return series of length T_eff = T_cov - 1.  The corresponding
  dates are dates[warmup + 1 : warmup + T_cov], stored as dates_offset = warmup + 1.

Weight convention
-----------------
  Eigenvectors from scipy SVD on a real symmetric PSD matrix are orthonormal
  (||w||_2 = 1).  Eigenportfolio returns are therefore r_k = w_k^T r, the
  projection of the return vector onto the unit eigenvector.  No additional
  L1 or sum-to-one normalisation is applied, consistent with Avellaneda &
  Lee (2010).  Sharpe ratios and cumulative P&L are directly comparable
  across PCs because all weight vectors share the same L2 norm.
"""

import os
import numpy as np
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Core return computation
# ---------------------------------------------------------------------------

def compute_eigenportfolio_returns(
    returns_matrix: np.ndarray,
    decomp_results: Dict[int, Dict],
    num_components: int,
) -> Dict[int, Dict]:
    """
    Compute daily out-of-sample returns for the top-K eigenportfolios.

    Parameters
    ----------
    returns_matrix : (T, N) array of log returns
    decomp_results : output of decompose_all() — maps half_life ->
                     {eigenvectors, eigenvalues, num_components,
                      absorption_ratios, warmup_steps}
    num_components : number of leading PCs to compute returns for (K).
                     Capped at N automatically so reducing the universe
                     never causes an index error.

    Returns
    -------
    dict : half_life -> {
        'ep_returns'   : (T_eff, K) float64 — daily eigenportfolio log-returns
        'dates_offset' : int — index in the full dates list of ep_returns[0]
    }
    """
    T, N = returns_matrix.shape
    K    = min(num_components, N)   # cap at universe size
    results: Dict[int, Dict] = {}

    for hl, decomp in decomp_results.items():
        warmup  = int(decomp["warmup_steps"])
        eigvecs = decomp["eigenvectors"]      # (T_cov, N, N), columns = PCs
        T_cov   = eigvecs.shape[0]
        T_eff   = T_cov - 1                   # one step shorter: need future return

        r_future = returns_matrix[warmup + 1 : warmup + T_eff + 1, :]  # (T_eff, N)

        ep_returns = np.empty((T_eff, K), dtype=np.float64)
        for k in range(K):
            w_series         = eigvecs[:T_eff, :, k]
            ep_returns[:, k] = np.sum(w_series * r_future, axis=1)

        results[hl] = {
            "ep_returns":   ep_returns,
            "dates_offset": warmup + 1,
        }

    return results


# ---------------------------------------------------------------------------
# Performance analytics
# ---------------------------------------------------------------------------

def compute_eigenportfolio_performance(
    ep_returns: np.ndarray,
    ann_factor: float = 252.0,
) -> Dict[str, np.ndarray]:
    """
    Annualised performance metrics for each eigenportfolio.

    Parameters
    ----------
    ep_returns : (T, K) daily eigenportfolio returns
    ann_factor : trading days per year

    Returns
    -------
    dict with:
        annualised_return  : (K,)  annualised mean return
        annualised_vol     : (K,)  annualised volatility  (ddof=1)
        sharpe             : (K,)  annualised Sharpe, zero risk-free rate
        max_drawdown       : (K,)  worst peak-to-trough  (≤ 0)
        cumulative_returns : (T, K) wealth index starting at 0.0
    """
    T, K = ep_returns.shape   # K derived from data, not config

    cumulative_returns = np.cumprod(1.0 + ep_returns, axis=0) - 1.0
    ann_ret = np.mean(ep_returns, axis=0) * ann_factor
    ann_vol = np.std(ep_returns,  axis=0, ddof=1) * np.sqrt(ann_factor)
    sharpe  = np.where(ann_vol > 1e-12, ann_ret / ann_vol, 0.0)

    max_dd = np.zeros(K, dtype=np.float64)
    for k in range(K):
        wealth    = np.cumprod(1.0 + ep_returns[:, k])
        peak      = np.maximum.accumulate(wealth)
        drawdown  = (wealth - peak) / np.where(peak > 1e-12, peak, 1.0)
        max_dd[k] = float(drawdown.min())

    return {
        "annualised_return":  ann_ret,
        "annualised_vol":     ann_vol,
        "sharpe":             sharpe,
        "max_drawdown":       max_dd,
        "cumulative_returns": cumulative_returns,
    }


# ---------------------------------------------------------------------------
# Sector loadings extraction (for visualisation)
# ---------------------------------------------------------------------------

def get_sector_loadings_timeseries(
    decomp_results: Dict[int, Dict],
    half_life: int,
    num_components: int,
) -> np.ndarray:
    """
    Extract rolling sector loadings (eigenvectors) for the top-K PCs.

    Returns
    -------
    loadings_ts : (T_cov, N, K) float64
    """
    eigvecs = decomp_results[half_life]["eigenvectors"]   # (T_cov, N, N)
    N       = eigvecs.shape[1]
    K       = min(num_components, N)
    return eigvecs[:, :, :K].copy()


def get_sector_loadings_snapshot(
    decomp_results: Dict[int, Dict],
    half_life: int,
    num_components: int,
) -> np.ndarray:
    """
    Return the most recent eigenvector matrix (final timestep only).

    Returns
    -------
    loadings : (N, K) — columns are PC1 … PCK as of the last date
    """
    eigvecs = decomp_results[half_life]["eigenvectors"]   # (T_cov, N, N)
    N       = eigvecs.shape[1]
    K       = min(num_components, N)
    return eigvecs[-1, :, :K].copy()


# ---------------------------------------------------------------------------
# CSV + export helpers
# ---------------------------------------------------------------------------

def build_eigenportfolio_dataframe(
    dates: List,
    ep_results: Dict[int, Dict],
    ep_perf_all: Dict[int, Dict],
) -> Dict[int, Dict]:
    """
    Assemble per-half-life DataFrames for CSV export (returns + cumulative P&L).
    K is derived from ep_returns.shape[1] — never passed as a parameter.

    Returns
    -------
    dict : half_life -> {
        'dates'   : list of date strings
        'returns' : (T_eff, K) ep_returns
        'cum_ret' : (T_eff, K) cumulative_returns
    }
    """
    out: Dict[int, Dict] = {}
    for hl, ep in ep_results.items():
        offset  = ep["dates_offset"]
        T_eff   = ep["ep_returns"].shape[0]   # K read from data, not config
        t_dates = [str(d)[:10] for d in dates[offset : offset + T_eff]]
        out[hl] = {
            "dates":   t_dates,
            "returns": ep["ep_returns"],
            "cum_ret": ep_perf_all[hl]["cumulative_returns"],
        }
    return out

def save_eigenvectors(
    decomp_results: Dict[int, Dict],
    output_dir: str,
    half_lives: Optional[List[int]] = None,
) -> None:
    """
    Save eigenvector arrays to .npy files.
    Called from main.py when config.eigenportfolio.save_eigenvectors = true.

    Files written: {output_dir}/eigvecs_hl{X}.npy for each half-life X.
    half_lives defaults to all keys in decomp_results if not provided.
    """
    os.makedirs(output_dir, exist_ok=True)
    hls = half_lives if half_lives is not None else sorted(decomp_results.keys())
    for hl in hls:
        eigvecs = decomp_results[hl]["eigenvectors"]   # (T_cov, N, N)
        path    = os.path.join(output_dir, f"eigvecs_hl{hl}.npy")
        np.save(path, eigvecs)
        print(f"   Saved {path}  shape={eigvecs.shape}")
