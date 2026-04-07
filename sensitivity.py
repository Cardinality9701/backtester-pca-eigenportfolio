# =============================================================
# sensitivity.py
# Strategy-aware parameter sensitivity heatmap.
#
# Sweeps (param_a, param_b) from src/grids.py for the strategy
# in the target run, runs one silent backtest per valid pair,
# and plots a Sharpe heatmap.
#
# Usage:
#   python sensitivity.py
#   python sensitivity.py --run outputs/tsmom_SPY_20260320_190000
# =============================================================
from __future__ import annotations

import os
import sys
import tempfile
import argparse
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import yaml

sys.path.insert(0, str(Path(__file__).parent))

# Shared helpers — avoids re-implementing find_latest_output_dir with
# the wrong (alphabetical) sort, and keeps require() in one place
from visualize import find_latest_output_dir
from run_backtest import fetch_data, require

from src.engine_analytics import run_backtest
from src.grids import get_grid, is_valid_pair, STRATEGY_GRIDS as GRIDS


# ---------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------

def load_run_config(out_dir: Path) -> Dict:
    config_path = out_dir / "config_snapshot.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config_snapshot.yaml found in {out_dir}. "
            "Run run_backtest.py first."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------
# Single silent backtest — thin wrapper around run_backtest()
# ---------------------------------------------------------------

def run_single(
    tickers: List[str],
    aux_tickers: List[str],
    start: str,
    end: str,
    data_dir: str,
    buffer_size: int,
    strategy_type: str,
    strategy_params: Dict,    # must NOT contain "type" key
    initial_capital: float,
    risk_fraction: float,
    commission_pct: float,
    slippage_bps: float,
    trading_days: int,
    db_path: str,
    regime_filter_cfg: Optional[dict] = None,
) -> float:
    """
    Runs one backtest silently and returns the annualised Sharpe ratio.
    Returns np.nan on any failure — grid search continues regardless.

    run_backtest() handles all component wiring internally.
    This function only handles the temp-DB lifecycle and Sharpe extraction.
    """
    if os.path.exists(db_path):
        os.remove(db_path)
    try:
        result = run_backtest(
            tickers=tickers,
            aux_tickers=aux_tickers,
            start=start,
            end=end,
            data_dir=data_dir,
            buffer_size=buffer_size,
            strategy_type=strategy_type,
            strategy_params=strategy_params,
            initial_capital=initial_capital,
            risk_fraction=risk_fraction,
            commission_pct=commission_pct,
            slippage_bps=slippage_bps,
            db_path=db_path,
            trading_days=trading_days,
            regime_filter_cfg=regime_filter_cfg,
        )
        # run_backtest() returns a pl.DataFrame — extract scalar Sharpe
        return float(result["sharpe_ratio"][0])
    except Exception as e:
        print(f" [warn] {e}")
        return np.nan
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


# ---------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------

def run_grid(
    tickers: List[str],
    aux_tickers: List[str],
    start: str,
    end: str,
    data_dir: str,
    buffer_size: int,
    strategy_type: str,
    base_params: Dict,        # already has "type" stripped
    grid: Dict,
    initial_capital: float,
    risk_fraction: float,
    commission_pct: float,
    slippage_bps: float,
    trading_days: int,
    regime_filter_cfg: Optional[dict] = None,
) -> Tuple[np.ndarray, List, List]:
    """
    Iterate over all valid (axis_a, axis_b) pairs, run one silent backtest
    per pair, and return the Sharpe matrix plus the axis value lists.
    """
    axis_a   = grid["axis_a"]
    axis_b   = grid["axis_b"]
    param_a  = grid["param_a"]
    param_b  = grid["param_b"]
    matrix   = np.full((len(axis_a), len(axis_b)), np.nan)

    total = sum(
        1 for a, b in product(axis_a, axis_b)
        if is_valid_pair(strategy_type, a, b)
    )
    done = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        for ai, a_val in enumerate(axis_a):
            for bi, b_val in enumerate(axis_b):
                if not is_valid_pair(strategy_type, a_val, b_val):
                    continue

                done += 1
                print(
                    f"  [{done:>3}/{total}]  "
                    f"{param_a}={a_val}  {param_b}={b_val} ...",
                    end=" ", flush=True,
                )

                # Override only the two swept params; all others from base_params
                params = {**base_params, param_a: a_val, param_b: b_val}
                db_path = os.path.join(tmp_dir, f"s_{ai}_{bi}.db")

                sharpe = run_single(
                    tickers=tickers,
                    aux_tickers=aux_tickers,
                    start=start,
                    end=end,
                    data_dir=data_dir,
                    buffer_size=buffer_size,
                    strategy_type=strategy_type,
                    strategy_params=params,
                    initial_capital=initial_capital,
                    risk_fraction=risk_fraction,
                    commission_pct=commission_pct,
                    slippage_bps=slippage_bps,
                    trading_days=trading_days,
                    db_path=db_path,
                    regime_filter_cfg=regime_filter_cfg,
                )
                matrix[ai][bi] = sharpe
                label = f"{sharpe:.3f}" if not np.isnan(sharpe) else "n/a"
                print(f"Sharpe = {label}")

    return matrix, axis_a, axis_b


# ---------------------------------------------------------------
# Heatmap plot
# ---------------------------------------------------------------

def plot_heatmap(
    matrix: np.ndarray,
    axis_a: List,
    axis_b: List,
    grid: Dict,
    out_dir: Path,
    tickers_str: str,
    strategy_name: str,
) -> None:
    DARK_BG  = "#0f0f0f"
    PANEL_BG = "#1a1a1a"
    TEXT     = "#d0d0d0"
    GRID_C   = "#2a2a2a"

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    valid = matrix[~np.isnan(matrix)]
    if len(valid) == 0:
        print("[sensitivity] No valid results to plot.")
        return

    vmin, vmax = float(valid.min()), float(valid.max())
    norm = (
        TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        if vmin < 0 < vmax
        else plt.Normalize(vmin=vmin, vmax=vmax)
    )

    im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto", origin="upper")

    ax.set_xticks(range(len(axis_b)))
    ax.set_xticklabels(axis_b, fontsize=9, color=TEXT)
    ax.set_yticks(range(len(axis_a)))
    ax.set_yticklabels(axis_a, fontsize=9, color=TEXT)
    ax.set_xlabel(grid["label_b"], fontsize=10, color=TEXT)
    ax.set_ylabel(grid["label_a"], fontsize=10, color=TEXT)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)

    for ai in range(len(axis_a)):
        for bi in range(len(axis_b)):
            val = matrix[ai, bi]
            if np.isnan(val):
                ax.text(bi, ai, "—", ha="center", va="center",
                        fontsize=8, color="#555555")
            else:
                brightness = (val - vmin) / (vmax - vmin + 1e-10)
                txt_color = "black" if 0.35 < brightness < 0.85 else TEXT
                ax.text(bi, ai, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=txt_color, fontweight="bold")

    # Highlight best cell with white border
    best_idx = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    ax.add_patch(plt.Rectangle(
        (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
        fill=False, edgecolor="white", lw=2.5, zorder=5,
    ))
    best_a = axis_a[best_idx[0]]
    best_b = axis_b[best_idx[1]]
    best_v = matrix[best_idx]

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Annualised Sharpe Ratio", color=TEXT, fontsize=9)
    cbar.ax.tick_params(colors=TEXT, labelsize=8)

    ax.set_title(
        f"Parameter Sensitivity — {tickers_str} {strategy_name}\n"
        f"Annualised Sharpe across ({grid['param_a']}, {grid['param_b']}) grid",
        fontsize=12, color=TEXT, pad=14,
    )

    param_a_label = grid["param_a"].replace("_", " ")
    param_b_label = grid["param_b"].replace("_", " ")
    fig.text(
        0.5, 0.01,
        f"Best: {param_a_label}={best_a}, {param_b_label}={best_b}  "
        f"→  Sharpe = {best_v:.3f}   (white border)     "
        f"Grey cells = invalid pairs",
        ha="center", va="bottom", fontsize=8.5,
        color=TEXT, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                  edgecolor=GRID_C, alpha=0.85),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_path = out_dir / "sensitivity_heatmap.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[sensitivity] Heatmap saved → {save_path}")


# ---------------------------------------------------------------
# CSV saver
# ---------------------------------------------------------------

def save_sensitivity_csv(
    matrix: np.ndarray,
    axis_a: List,
    axis_b: List,
    grid: Dict,
    out_dir: Path,
) -> None:
    param_a = grid["param_a"]
    param_b = grid["param_b"]
    rows = [f"{param_a},{param_b},sharpe"]
    for ai, a_val in enumerate(axis_a):
        for bi, b_val in enumerate(axis_b):
            val = matrix[ai, bi]
            sharpe_str = f"{val:.6f}" if not np.isnan(val) else ""
            rows.append(f"{a_val},{b_val},{sharpe_str}")
    path = out_dir / "sensitivity_results.csv"
    path.write_text("\n".join(rows))
    print(f"[sensitivity] Results CSV → {path}")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strategy-aware parameter sensitivity heatmap."
    )
    parser.add_argument("--run",     type=str,   default=None)
    parser.add_argument("--capital", type=float, default=None)  # None → config wins
    args = parser.parse_args()

    out_dir = Path(args.run) if args.run else find_latest_output_dir()
    cfg     = load_run_config(out_dir)

    # ── Config extraction ──────────────────────────────────────────────────
    cfg_data      = require(cfg, "data",      "root")
    cfg_strategy  = require(cfg, "strategy",  "root")
    cfg_risk      = require(cfg, "risk",       "root")
    cfg_exec      = require(cfg, "execution",  "root")
    cfg_analytics = require(cfg, "analytics",  "root")
    cfg_filter = cfg.get("regime_filter")

    # Backward-compatibility: support old `ticker: "SPY"` single-string format
    tickers = cfg_data.get("tickers") or cfg_data.get("ticker")
    if tickers is None:
        print("[error] Missing required config key: data.tickers")
        sys.exit(1)
    if isinstance(tickers, str):
        tickers = [tickers]
    aux_tickers: List[str] = cfg_data.get("aux_tickers", [])

    start        = require(cfg_data,      "start",                  "data")
    end          = require(cfg_data,      "end",                    "data")
    buffer_size  = int(require(cfg_data,  "buffer_size",            "data"))
    data_dir     = cfg_data.get("data_dir", "data")

    strategy_type   = require(cfg_strategy,  "type",                "strategy")
    # Strip "type" — _build_strategy() must not receive it as a kwarg
    base_params     = {k: v for k, v in cfg_strategy.items() if k != "type"}

    initial_capital = float(require(cfg_risk,      "initial_capital",       "risk"))
    risk_fraction   = float(require(cfg_risk,      "risk_fraction",         "risk"))
    commission_pct  = float(require(cfg_exec,      "commission_pct",        "execution"))
    slippage_bps    = float(require(cfg_exec,      "slippage_bps",          "execution"))
    trading_days    = int(require(cfg_analytics,   "trading_days_per_year", "analytics"))

    if args.capital is not None:
        initial_capital = args.capital

    if strategy_type not in GRIDS:
        print(f"[error] No sensitivity grid defined for: '{strategy_type}'")
        print(f"        Available: {list(GRIDS.keys())}")
        sys.exit(1)

    grid        = get_grid(strategy_type)
    tickers_str = ", ".join(tickers)
    n_valid     = sum(
        1 for a, b in product(grid["axis_a"], grid["axis_b"])
        if is_valid_pair(strategy_type, a, b)
    )

    # Ensure all tickers are cached locally before grid search begins
    for t in tickers + aux_tickers:
        fetch_data(ticker=t, start=start, end=end, data_dir=Path(data_dir))

    print(f"\n{'='*54}")
    print(f"  Sensitivity Analysis — {tickers_str} {strategy_type}")
    print(f"  {grid['param_a']} axis : {grid['axis_a']}")
    print(f"  {grid['param_b']} axis : {grid['axis_b']}")
    print(f"  Valid pairs           : {n_valid}")
    print(f"  Output dir            : {out_dir}")
    print(f"{'='*54}\n")

    matrix, axis_a, axis_b = run_grid(
        tickers=tickers,
        aux_tickers=aux_tickers,
        start=start,
        end=end,
        data_dir=data_dir,
        buffer_size=buffer_size,
        strategy_type=strategy_type,
        base_params=base_params,
        grid=grid,
        initial_capital=initial_capital,
        risk_fraction=risk_fraction,
        commission_pct=commission_pct,
        slippage_bps=slippage_bps,
        trading_days=trading_days,
        regime_filter_cfg=cfg_filter,
    )

    save_sensitivity_csv(matrix, axis_a, axis_b, grid, out_dir)
    plot_heatmap(matrix, axis_a, axis_b, grid, out_dir, tickers_str, strategy_type)


if __name__ == "__main__":
    main()
