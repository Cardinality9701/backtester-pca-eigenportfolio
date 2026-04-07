"""
run_eigenportfolio.py — PCA Correlation Regime Pipeline
==========================================
Orchestrates the full pipeline end-to-end:

  1. Preprocess data (download, align, log-returns, VIX)
  2. Compute rolling EWM covariance matrices (all half-lives)
  3. Spectral decomposition (PCA / SVD)
  4. Risk monitoring (absorption ratio, delta-AR, champion vs challenger)
  5. Eigenportfolio returns & performance        ← Phase 1
  6. Ledoit-Wolf covariance series               ← Phase 2
  7. Predictive analysis: delta-AR vs VIX        ← Phase 3
  8. Persist results to CSV / npy
  9. Generate figures

Usage:
  python main.py
  python main.py --config path/to/config.yaml

Data split discipline
---------------------
  The train / val / test boundaries in config.yaml are for Phase 4 strategy
  evaluation.  Phases 1-3 compute rolling statistics over the FULL date range
  so that figures are visually complete.  No forward-looking bias is introduced
  because every estimator at time t uses only data up to t.
  The test set (2019-01-01 → end_date) must NOT be used for strategy selection
  or parameter tuning until Phase 4.
"""

import argparse
import os

import numpy as np
import polars as pl
import yaml

from eigenportfolio.data_preprocessor import preprocess
from eigenportfolio.covariance_engine import (
    compute_all_covariance_series,
    compute_ledoit_wolf_full,
)
from eigenportfolio.spectral_decomposer import decompose_all
from eigenportfolio.risk_monitor        import run_risk_monitor
from eigenportfolio.eigenportfolio      import (
    compute_eigenportfolio_returns,
    compute_eigenportfolio_performance,
    save_eigenvectors,
)
from eigenportfolio.predictive_analysis import run_predictive_analysis
from eigenportfolio.visualizer import run_all as run_visualizations
from eigenportfolio.regime_classifier  import run_pc_selection
from eigenportfolio.signal_generator   import run_signal_generator
from eigenportfolio.backtester         import run_backtester
from eigenportfolio.performance_evaluator import run_performance_evaluator
from eigenportfolio.regime_visualizer     import run_regime_visualizations
from eigenportfolio.data_preprocessor import get_split_masks, build_aligned_panel
from eigenportfolio.regime_classifier import run_pc_selection

# ---------------------------------------------------------------------------
# CSV / npy persistence helpers
# ---------------------------------------------------------------------------

def save_results_csv(
    dates,
    decomp_results: dict,
    monitor_results: dict,
    output_dir: str,
) -> None:
    """Persist rolling time-series results for each half-life to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    for hl in sorted(decomp_results.keys()):
        warmup  = int(decomp_results[hl]["warmup_steps"])
        t_dates = dates[warmup:]
        rows = {
            "date":             [str(d)[:10] for d in t_dates],
            "absorption_ratio": decomp_results[hl]["absorption_ratios"].tolist(),
            "num_components":   decomp_results[hl]["num_components"].tolist(),
            "pca_explained":    monitor_results[hl]["pca_explained"].tolist(),
            "ew_explained":     monitor_results[hl]["ew_explained"].tolist(),
            "pca_edge":         monitor_results[hl]["pca_edge"].tolist(),
            "delta_ar":         monitor_results[hl]["delta_ar"].tolist(),
        }
        fname = os.path.join(output_dir, f"results_hl{hl}.csv")
        pl.DataFrame(rows).write_csv(fname)
        print(f"   Saved {fname} ({len(t_dates)} rows)")


def save_eigenportfolio_csv(
    dates,
    ep_results: dict,
    ep_perf_all: dict,
    output_dir: str,
) -> None:
    """
    Persist eigenportfolio returns and cumulative P&L to CSV.
    One file per half-life: ep_hl{X}.csv
    K is derived from ep_returns.shape[1] — never passed as a parameter.
    Columns: date, pc1_ret … pcK_ret, pc1_cum … pcK_cum
    """
    os.makedirs(output_dir, exist_ok=True)
    for hl in sorted(ep_results.keys()):
        ep     = ep_results[hl]
        perf   = ep_perf_all[hl]
        offset = ep["dates_offset"]
        T_eff  = ep["ep_returns"].shape[0]
        K      = ep["ep_returns"].shape[1]   # derived from data

        t_dates = [str(d)[:10] for d in dates[offset : offset + T_eff]]
        rows: dict = {"date": t_dates}
        for k in range(K):
            rows[f"pc{k + 1}_ret"] = ep["ep_returns"][:, k].tolist()
            rows[f"pc{k + 1}_cum"] = perf["cumulative_returns"][:, k].tolist()

        fname = os.path.join(output_dir, f"ep_hl{hl}.csv")
        pl.DataFrame(rows).write_csv(fname)
        print(f"   Saved {fname} ({T_eff} rows, K={K})")


def save_ledoit_wolf(
    dates,
    cov_lw_series: np.ndarray,
    shrinkage_series: np.ndarray,
    lw_warmup: int,
    output_dir: str,
) -> None:
    """
    Persist LW outputs.

    cov_lw_series    → results/cov_lw_series.npy   (binary; loaded by horse_race.py)
    shrinkage_series → results/lw_shrinkage.csv    (human-readable diagnostic)
    """
    os.makedirs(output_dir, exist_ok=True)

    npy_path = os.path.join(output_dir, "cov_lw_series.npy")
    np.save(npy_path, cov_lw_series)
    print(f"   Saved {npy_path}  shape={cov_lw_series.shape}")

    t_dates  = [str(d)[:10] for d in dates[lw_warmup : lw_warmup + len(shrinkage_series)]]
    csv_path = os.path.join(output_dir, "lw_shrinkage.csv")
    pl.DataFrame({"date": t_dates, "shrinkage": shrinkage_series.tolist()}).write_csv(csv_path)
    print(f"   Saved {csv_path}  (ρ mean={shrinkage_series.mean():.4f}  "
          f"min={shrinkage_series.min():.4f}  max={shrinkage_series.max():.4f})")


def save_predictive_csv(
    predictive_results: dict,
    output_dir: str,
) -> None:
    """Persist rolling OLS R² series to CSV. One file per horizon."""
    os.makedirs(output_dir, exist_ok=True)
    rolling = predictive_results["rolling_r2"]
    for h, (h_dates, r2_arr) in rolling.items():
        rows = {
            "date":       [str(d)[:10] for d in h_dates],
            "rolling_r2": r2_arr.tolist(),
        }
        fname = os.path.join(output_dir, f"predictive_rolling_r2_h{h}.csv")
        pl.DataFrame(rows).write_csv(fname)
        print(f"   Saved {fname} ({len(h_dates)} rows, h={h}d)")


# ---------------------------------------------------------------------------
# Stdout summary helpers
# ---------------------------------------------------------------------------

def _print_ep_performance(ep_perf_all: dict) -> None:
    """K derived from data — no num_components parameter."""
    K = ep_perf_all[sorted(ep_perf_all.keys())[0]]["sharpe"].shape[0]
    print("\n   Eigenportfolio performance summary (full-sample):")
    header = f"   {'HL':>5}  " + "".join(
        f"  {'PC' + str(k + 1) + ' SR':>8}  {'Ann%':>7}  {'MDD%':>7}"
        for k in range(K)
    )
    print(header)
    for hl in sorted(ep_perf_all.keys()):
        p   = ep_perf_all[hl]
        row = f"   {hl:>5}d "
        for k in range(K):
            row += (
                f"  {p['sharpe'][k]:>+8.2f}"
                f"  {p['annualised_return'][k] * 100:>+7.1f}"
                f"  {p['max_drawdown'][k] * 100:>7.1f}"
            )
        print(row)
    print()


def _print_lw_summary(
    cov_lw_series: np.ndarray,
    shrinkage_series: np.ndarray,
    lw_warmup: int,
    dates: list,
) -> None:
    """
    Diagnostic summary for the Ledoit-Wolf shrinkage series.
    Top-10 highest-shrinkage dates should cluster around known stress events.
    """
    print(f"\n   Ledoit-Wolf summary:")
    print(f"   Shape        : {cov_lw_series.shape}")
    print(f"   Warmup       : {lw_warmup} days")
    print(f"   Shrinkage ρ  : mean={shrinkage_series.mean():.4f}  "
          f"std={shrinkage_series.std():.4f}  "
          f"min={shrinkage_series.min():.4f}  "
          f"max={shrinkage_series.max():.4f}")
    top10_idx   = np.argsort(shrinkage_series)[-10:][::-1]
    top10_dates = [str(dates[lw_warmup + int(i)])[:10] for i in top10_idx]
    top10_rho   = shrinkage_series[top10_idx]
    print("   Top-10 ρ dates (expect Lehman 2008 / COVID 2020 / rate-hike 2022):")
    for rank, (d, rho) in enumerate(zip(top10_dates, top10_rho), start=1):
        print(f"     {rank:2}. {d}   ρ = {rho:.4f}")
    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(open(config_path))

    results_dir        = cfg["output"]["results_dir"]
    half_lives         = cfg["ewm"]["half_lives"]
    warmup_multiplier  = cfg["ewm"]["warmup_multiplier"]
    variance_threshold = cfg["pca"]["variance_threshold"]
    short_win          = cfg["risk_monitor"]["absorption_ratio_short_window"]
    long_win           = cfg["risk_monitor"]["absorption_ratio_long_window"]

    # All .get() defaults removed — KeyError on missing key is intentional
    ep_cfg         = cfg["eigenportfolio"]
    num_components = int(ep_cfg["num_components"])
    ann_factor     = float(ep_cfg["ann_factor"])
    save_eigvecs   = bool(ep_cfg.get("save_eigenvectors", False))

    lw_cfg      = cfg["ledoit_wolf"]
    lw_window   = int(lw_cfg["window"])
    lw_centered = bool(lw_cfg["assume_centered"])

    vix_cfg = cfg["vix"]
    vix_hl = int(ep_cfg["halflife_primary"])   # must exist in ewm.half_lives

    # ---- Step 1: Preprocess ------------------------------------------------
    print("[1/10] Downloading and preprocessing data...")
    df, returns_matrix, tickers, vix_series = preprocess(config_path)
    dates = df["date"].to_list()

    split_masks = get_split_masks(dates, cfg["splits"])   # ← new
    print(f"   Train: {split_masks['train'].sum()}d  "
          f"Val: {split_masks['val'].sum()}d  "
          f"Test: {split_masks['test'].sum()}d")

    N     = returns_matrix.shape[1]
    print(f"   Universe : {tickers}  (N={N})")
    print(f"   Shape    : {returns_matrix.shape} (T x N)")
    print(f"   VIX      : {vix_series.shape}  mean={vix_series.mean():.1f}  "
          f"max={vix_series.max():.1f}")

    # ---- Step 2: Rolling EWM Covariance ------------------------------------
    print("[2/10] Computing rolling EWM covariance matrices...")
    cov_results = compute_all_covariance_series(
        returns_matrix, half_lives, warmup_multiplier
    )
    for hl, (cov_series, ws) in cov_results.items():
        print(f"   HL={hl:>3}d : cov_series shape={cov_series.shape}  warmup={ws}")

    # ---- Step 3: Spectral Decomposition ------------------------------------
    print("[3/10] Running spectral decomposition (SVD)...")
    decomp_results = decompose_all(cov_results, variance_threshold)
    for hl, d in decomp_results.items():
        median_k = int(np.median(d["num_components"]))
        mean_ar  = float(np.nanmean(d["absorption_ratios"]))
        print(f"   HL={hl:>3}d : median K={median_k}  mean AR={mean_ar:.3f}")

    # ---- Step 4: Risk Monitor ----------------------------------------------
    print("[4/10] Computing absorption ratio anomaly and champion vs challenger...")
    monitor_results = run_risk_monitor(
        decomp_results, cov_results, short_win, long_win
    )

    # ---- Step 5: Eigenportfolio Returns & Performance  (Phase 1) ----------
    print("[5/10] Computing eigenportfolio returns and performance (Phase 1)...")
    ep_results  = compute_eigenportfolio_returns(
        returns_matrix, decomp_results, num_components
    )
    ep_perf_all: dict = {}
    for hl in sorted(ep_results.keys()):
        ep_perf_all[hl] = compute_eigenportfolio_performance(
            ep_results[hl]["ep_returns"], ann_factor=ann_factor
        )
        T_eff = ep_results[hl]["ep_returns"].shape[0]
        K_eff = ep_results[hl]["ep_returns"].shape[1]
        print(f"   HL={hl:>3}d : T_eff={T_eff}  K={K_eff}  "
              f"dates_offset={ep_results[hl]['dates_offset']}")
    _print_ep_performance(ep_perf_all)

    # ---- Step 6: Ledoit-Wolf Covariance  (Phase 2) -------------------------
    print("[6/10] Computing Ledoit-Wolf covariance series (Phase 2)...")
    cov_lw_series, shrinkage_series, lw_warmup = compute_ledoit_wolf_full(
        returns_matrix, window=lw_window, assume_centered=lw_centered
    )
    _print_lw_summary(cov_lw_series, shrinkage_series, lw_warmup, dates)

    # ---- Step 7: Predictive Analysis  (Phase 3) ----------------------------
    print("[7/10] Running predictive analysis: delta-AR vs VIX (Phase 3)...")
    vix_warmup         = int(decomp_results[vix_hl]["warmup_steps"])
    vix_series_aligned = vix_series[vix_warmup:]
    delta_ar_series    = monitor_results[vix_hl]["delta_ar"]
    dates_aligned      = dates[vix_warmup:]
    predictive_results = run_predictive_analysis(
        delta_ar   = delta_ar_series,
        vix_series = vix_series_aligned,
        dates      = dates_aligned,
        cfg        = vix_cfg,
    )

    # ---- Step 8: Save results ----------------------------------------------
    print("[8/10] Saving results...")
    save_results_csv(dates, decomp_results, monitor_results, results_dir)
    save_eigenportfolio_csv(dates, ep_results, ep_perf_all, results_dir)
    save_ledoit_wolf(dates, cov_lw_series, shrinkage_series, lw_warmup, results_dir)
    save_predictive_csv(predictive_results, results_dir)
    if save_eigvecs:
        save_eigenvectors(decomp_results, results_dir)

    # ---- Step 9: Visualise -------------------------------------------------
    print("[9/10] Generating figures...")
    run_visualizations(
        dates,
        decomp_results,
        monitor_results,
        output_dir         = results_dir,
        ep_results         = ep_results,
        ep_perf_all        = ep_perf_all,
        tickers            = tickers,
        ep_config          = ep_cfg,
        predictive_results = predictive_results,
    )

    # ---- Step 10: Regime Strategies (Phase 4) ----------------------------------
    print("[10/10] Running PC selection and regime labelling (Phase 4)...")

    primary_hl  = cfg["eigenportfolio"]["halflife_primary"]
    warmup      = decomp_results[primary_hl]["warmup_steps"]
    rs_cfg      = cfg["regime_strats"]

    panel       = build_aligned_panel(dates, ep_results, monitor_results,
                                      hl=primary_hl, warmup=warmup)
    split_masks = get_split_masks(panel["date"].to_list(), cfg["splits"])

    result, labelled_panel = run_pc_selection(panel, split_masks, rs_cfg)
    signal_df              = run_signal_generator(labelled_panel, split_masks,
                                              result, rs_cfg, cfg["eigenportfolio"])
    backtest_df            = run_backtester(signal_df, split_masks, result,
                                        rs_cfg, cfg["eigenportfolio"])
    perf_results, summary_df, cum_ret_dict = run_performance_evaluator(
    backtest_df, split_masks, rs_cfg, cfg["eigenportfolio"]
    )

    summary_df.write_csv(
    os.path.join(results_dir, "regime_strats_performance.csv")
    )
    run_regime_visualizations(
    backtest_df, cum_ret_dict, perf_results,
    split_masks, result, rs_cfg, ep_cfg,
    output_dir=results_dir,
    )

    # ---- Completion ------------------------------------------------------------
    print(f"\nPipeline complete. All outputs in ./{results_dir}/")
    print(f"   LW covariance array : {results_dir}/cov_lw_series.npy  "
          f"(np.load() in Phase 5)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA Correlation Regime Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
