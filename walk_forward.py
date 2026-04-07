# =============================================================
# walk_forward.py
# Anchored OR rolling walk-forward validation.
# All parameters read from config_snapshot.yaml — no hardcoded
# research values anywhere in this file.
#
#   walk_forward:
#     method: "anchored"   # or "rolling"
#     oos_months: 12
#     is_months: 48
#     folds: 5
# =============================================================
from __future__ import annotations

import argparse
import csv as csv_module
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.ticker as mticker
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import yaml

sys.path.insert(0, str(Path(__file__).parent))

# Shared helpers — find_latest_output_dir must sort by st_mtime not alphabetically
from visualize import find_latest_output_dir
from run_backtest import fetch_data, require

from src.engine_analytics import run_backtest
from src.grids import get_grid, get_is_windows, is_valid_pair


# ---------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------

def load_run_config(out_dir: Path) -> Dict:
    config_path = out_dir / "config_snapshot.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------

def load_full_csv(csv_path: Path) -> Tuple[List[str], List[Dict]]:
    """
    Load all rows from the primary ticker CSV.
    Normalises the date column to "date" — fetch_data() saves as "date"
    but old cached files may have "timestamp". Normalising here means
    write_fold_data() always writes consistent column names for
    CSVParquetDataHandler to find.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    # Normalize "timestamp" → "date" for backward compatibility
    if "timestamp" in rows[0] and "date" not in rows[0]:
        rows = [
            {("date" if k == "timestamp" else k): v for k, v in r.items()}
            for r in rows
        ]

    rows.sort(key=lambda r: str(r["date"])[:10])
    dates = [str(r["date"])[:10] for r in rows]
    return dates, rows


def write_fold_data(
    fold_rows_by_ticker: Dict[str, List[Dict]],
    base_tmp: str,
    fold_idx: int,
    phase: str,
    aux_tickers: List[str],
) -> str:
    """
    Write IS or OOS subset rows for ALL primary tickers into a single
    data_dir. The original version only wrote tickers[0] and copied
    tickers[1:] from full-history cache — this caused look-ahead
    contamination in multi-ticker walk-forward runs because tickers[1:]
    would contain bars from outside the IS/OOS window.

    aux_tickers are still copied in full from data/ cache. They are
    regime signals (VIX etc.), not traded instruments, so full-history
    access does not constitute look-ahead bias.

    Returns the data_dir path string for passing to run_silent_wf().
    """
    data_dir = os.path.join(base_tmp, f"fold_{fold_idx}_{phase}")
    os.makedirs(data_dir, exist_ok=True)

    # Write date-subset rows for every primary ticker
    for ticker, rows in fold_rows_by_ticker.items():
        if not rows:
            raise ValueError(
                f"Empty fold rows for ticker '{ticker}' "
                f"in fold {fold_idx} phase '{phase}'. "
                "Check that all tickers have data covering this date range."
            )
        path = os.path.join(data_dir, f"{ticker}.csv")
        with open(path, "w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # Copy full-history aux tickers — safe because they are never traded
    for at in aux_tickers:
        src = Path("data") / f"{at}.csv"
        dst = os.path.join(data_dir, f"{at}.csv")
        if src.exists() and not os.path.exists(dst):
            shutil.copy(str(src), dst)

    return data_dir


def add_months(date_str: str, n: int) -> str:
    dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    month = dt.month - 1 + n
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, [31, 28, 29, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return datetime(year, month, day).strftime("%Y-%m-%d")


# ---------------------------------------------------------------
# Silent fold backtest — thin wrapper around run_backtest()
# ---------------------------------------------------------------

def run_silent_wf(
    data_dir: str,
    rows: List[Dict],
    tickers: List[str],
    aux_tickers: List[str],
    strategy_type: str,
    strategy_params: Dict,    # must NOT contain "type"
    initial_capital: float,
    risk_fraction: float,
    commission_pct: float,
    slippage_bps: float,
    trading_days: int,
    buffer_size: int,
    db_path: str,
    regime_filter_cfg: Optional[dict] = None
) -> Tuple[float, np.ndarray]:
    """
    Thin wrapper around run_backtest() for walk-forward fold backtests.

    start/end are derived from the rows themselves — the subset CSV in
    data_dir only contains the IS or OOS window, so passing its actual
    date range ensures CSVParquetDataHandler loads all rows without filtering
    any out.

    Returns (sharpe, equities) — equities are needed by stitch_oos_equity()
    and are read from the DB before cleanup.
    """
    start = str(rows[0]["date"])[:10]
    end   = str(rows[-1]["date"])[:10]

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
            regime_filter_cfg=regime_filter_cfg
        )
        sharpe = float(result["sharpe_ratio"][0])

        # DB still exists at this point — read equity curve for stitching
        conn = sqlite3.connect(db_path)
        equities = np.array(
            [r[0] for r in conn.execute(
                "SELECT equity FROM equity_curve ORDER BY id"
            ).fetchall()],
            dtype=float,
        )
        conn.close()
        return sharpe, equities

    except Exception as e:
        print(f" [warn] {e}")
        return np.nan, np.array([initial_capital], dtype=float)

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


# ---------------------------------------------------------------
# IS optimisation
# ---------------------------------------------------------------

def optimise_is(
    is_data_dir: str,
    is_rows: List[Dict],
    tickers: List[str],
    aux_tickers: List[str],
    strategy_type: str,
    base_params: Dict,          # "type" already stripped
    fast_windows: List,
    slow_windows: List,
    initial_capital: float,
    tmp_dir: str,
    fold_idx: int,
    commission_pct: float,
    slippage_bps: float,
    risk_fraction: float,
    trading_days: int,
    buffer_size: int,
    regime_filter_cfg: Optional[dict] = None
) -> Tuple[Any, Any, float]:
    """
    Grid search over (fast_windows × slow_windows) on IS data.
    Uses param_a / param_b from grids.py to set the correct kwargs —
    avoids hardcoding fast_window/lookback_window/window etc. here.
    fast → param_a, slow → param_b (consistent with get_is_windows()).
    """
    grid    = get_grid(strategy_type)
    param_a = grid["param_a"]
    param_b = grid["param_b"]

    best_sharpe = -np.inf
    best_fast   = fast_windows[0]
    best_slow   = slow_windows[-1]

    for fast in fast_windows:
        for slow in slow_windows:
            if not is_valid_pair(strategy_type, fast, slow):
                continue

            params = {**base_params, param_a: fast, param_b: slow}
            db = os.path.join(tmp_dir, f"opt_{fold_idx}_{fast}_{slow}.db")
            sharpe, _ = run_silent_wf(
                data_dir=is_data_dir,
                rows=is_rows,
                tickers=tickers,
                aux_tickers=aux_tickers,
                strategy_type=strategy_type,
                strategy_params=params,
                initial_capital=initial_capital,
                risk_fraction=risk_fraction,
                commission_pct=commission_pct,
                slippage_bps=slippage_bps,
                trading_days=trading_days,
                buffer_size=buffer_size,
                db_path=db,
                regime_filter_cfg=regime_filter_cfg,
            )
            if not np.isnan(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_fast, best_slow = fast, slow

    return best_fast, best_slow, best_sharpe


# ---------------------------------------------------------------
# Fold builders — anchored and rolling
# ---------------------------------------------------------------

def build_folds_anchored(
    dates: List[str],
    n_folds: int,
    oos_months: int,
) -> List[Dict]:
    start_date   = dates[0]
    oos_end_date = dates[-1]
    boundaries: List[Tuple[str, str]] = []

    for _ in range(n_folds):
        oos_start = add_months(oos_end_date, -oos_months)
        if oos_start <= start_date:
            break
        boundaries.append((oos_start, oos_end_date))
        oos_end_date = oos_start

    boundaries = list(reversed(boundaries))
    return _boundaries_to_folds(dates, boundaries, is_start_fixed=dates[0])


def build_folds_rolling(
    dates: List[str],
    n_folds: int,
    oos_months: int,
    is_months: int,
) -> List[Dict]:
    oos_end_date = dates[-1]
    boundaries: List[Tuple[str, str, str]] = []

    for _ in range(n_folds):
        oos_start = add_months(oos_end_date, -oos_months)
        is_start  = add_months(oos_start,    -is_months)
        if is_start < dates[0]:
            break
        boundaries.append((is_start, oos_start, oos_end_date))
        oos_end_date = oos_start

    boundaries = list(reversed(boundaries))
    folds = []
    for is_start, oos_start, oos_end in boundaries:
        is_idx  = [i for i, d in enumerate(dates) if is_start  <= d < oos_start]
        oos_idx = [i for i, d in enumerate(dates) if oos_start <= d <= oos_end]
        if len(is_idx) < 100 or len(oos_idx) < 10:
            continue
        folds.append({
            "is_start":    is_start,
            "is_end":      dates[is_idx[-1]],
            "oos_start":   oos_start,
            "oos_end":     oos_end,
            "is_indices":  is_idx,
            "oos_indices": oos_idx,
        })
    return folds


def _boundaries_to_folds(
    dates: List[str],
    boundaries: List[Tuple[str, str]],
    is_start_fixed: str,
) -> List[Dict]:
    folds = []
    for oos_start, oos_end in boundaries:
        is_idx  = [i for i, d in enumerate(dates) if is_start_fixed <= d < oos_start]
        oos_idx = [i for i, d in enumerate(dates) if oos_start      <= d <= oos_end]
        if len(is_idx) < 100 or len(oos_idx) < 10:
            continue
        folds.append({
            "is_start":    is_start_fixed,
            "is_end":      dates[is_idx[-1]],
            "oos_start":   oos_start,
            "oos_end":     oos_end,
            "is_indices":  is_idx,
            "oos_indices": oos_idx,
        })
    return folds


def build_folds(
    dates: List[str],
    n_folds: int,
    oos_months: int,
    method: str,
    is_months: int,
) -> List[Dict]:
    """Dispatcher — all parameters must be passed explicitly, no defaults."""
    if method == "rolling":
        return build_folds_rolling(dates, n_folds, oos_months, is_months)
    elif method == "anchored":
        return build_folds_anchored(dates, n_folds, oos_months)
    else:
        raise ValueError(
            f"Unknown walk_forward method: '{method}'. "
            "Choose 'anchored' or 'rolling'."
        )


# ---------------------------------------------------------------
# Main walk-forward runner
# ---------------------------------------------------------------

def run_walk_forward(
    csv_path: Path,
    tickers: List[str],
    aux_tickers: List[str],
    strategy_type: str,
    base_params: Dict,
    fast_windows: List,
    slow_windows: List,
    initial_capital: float,
    n_folds: int,
    oos_months: int,
    method: str,
    is_months: int,
    commission_pct: float,
    slippage_bps: float,
    risk_fraction: float,
    trading_days: int,
    buffer_size: int,
    tmp_dir: str,
    regime_filter_cfg: Optional[Dict] = None,
) -> List[Dict]:
    """
    Main walk-forward loop.

    FIXED (multi-ticker): All tickers are loaded upfront and subset
    to IS/OOS windows before being written to the fold data_dir.
    Previously only tickers[0] was subset — tickers[1:] were copied
    from full-history cache, contaminating OOS folds with future data.

    tickers[0] drives fold boundary dates. All other tickers are
    filtered to the same date indices — rows are matched by position,
    not by date string, so sparse calendars (e.g. EEM holidays) stay
    correctly aligned after CSVParquetDataHandler inner-joins on dates.
    """
    ticker = tickers[0]
    dates, all_rows = load_full_csv(csv_path)

    # Pre-load ALL tickers upfront — one pass, no repeated disk I/O
    # in the fold loop. Keyed by ticker name for O(1) lookup.
    all_rows_by_ticker: Dict[str, List[Dict]] = {ticker: all_rows}
    for t in tickers[1:]:
        t_path = Path("data") / f"{t}.csv"
        if not t_path.exists():
            raise FileNotFoundError(
                f"No cached data for '{t}' at {t_path}. "
                "Run run_backtest.py first to download and cache all tickers."
            )
        _, t_rows = load_full_csv(t_path)
        all_rows_by_ticker[t] = t_rows

    folds = build_folds(dates, n_folds, oos_months, method, is_months)

    if not folds:
        raise RuntimeError(
            f"Could not build any valid folds using method='{method}'. "
            "Try reducing folds/oos_months or increasing is_months."
        )

    grid    = get_grid(strategy_type)
    param_a = grid["param_a"]
    param_b = grid["param_b"]

    print(f"  Method : {method.upper()}")
    print(f"  Built  : {len(folds)} folds  ({oos_months}-month OOS windows)")
    print(f"  Tickers: {tickers}\n")

    results = []
    for idx, fold in enumerate(folds):
        print(f"  {'─'*46}")
        print(f"  Fold {idx + 1}/{len(folds)}")
        print(f"    IS  : {fold['is_start']} → {fold['is_end']}"
              f"  ({len(fold['is_indices'])} bars)")
        print(f"    OOS : {fold['oos_start']} → {fold['oos_end']}"
              f"  ({len(fold['oos_indices'])} bars)")

        # FIXED: subset ALL tickers to IS/OOS windows using fold indices
        # from the primary ticker's date array. Tickers with different
        # calendar coverage may have fewer rows — guard with min(i, len-1)
        # rather than silently dropping tickers.
        is_rows_by_ticker: Dict[str, List[Dict]] = {}
        oos_rows_by_ticker: Dict[str, List[Dict]] = {}

        for t in tickers:
            t_all = all_rows_by_ticker[t]
            t_len = len(t_all)

            is_rows_by_ticker[t] = [
                t_all[i] for i in fold["is_indices"] if i < t_len
            ]
            oos_rows_by_ticker[t] = [
                t_all[i] for i in fold["oos_indices"] if i < t_len
            ]

            # Warn — don't silently fail — if a ticker has sparse coverage
            expected_is  = len(fold["is_indices"])
            expected_oos = len(fold["oos_indices"])
            actual_is    = len(is_rows_by_ticker[t])
            actual_oos   = len(oos_rows_by_ticker[t])

            if actual_is < expected_is * 0.9:
                print(f"    [warn] '{t}' IS rows: {actual_is}/{expected_is} "
                      f"({100*actual_is/expected_is:.0f}%). "
                      "Possible data gap — check CSV coverage.")
            if actual_oos < expected_oos * 0.9:
                print(f"    [warn] '{t}' OOS rows: {actual_oos}/{expected_oos} "
                      f"({100*actual_oos/expected_oos:.0f}%). "
                      "Possible data gap — check CSV coverage.")

        # Write correctly subset data_dirs for IS and OOS
        is_data_dir = write_fold_data(
            fold_rows_by_ticker=is_rows_by_ticker,
            base_tmp=tmp_dir,
            fold_idx=idx,
            phase="is",
            aux_tickers=aux_tickers,
        )
        oos_data_dir = write_fold_data(
            fold_rows_by_ticker=oos_rows_by_ticker,
            base_tmp=tmp_dir,
            fold_idx=idx,
            phase="oos",
            aux_tickers=aux_tickers,
        )

        # IS rows for run_silent_wf start/end derivation — use primary ticker
        is_rows_primary  = is_rows_by_ticker[ticker]
        oos_rows_primary = oos_rows_by_ticker[ticker]

        print(f"    Optimising IS parameters...", end=" ", flush=True)
        best_fast, best_slow, is_sharpe = optimise_is(
            is_data_dir=is_data_dir,
            is_rows=is_rows_primary,
            tickers=tickers,
            aux_tickers=aux_tickers,
            strategy_type=strategy_type,
            base_params=base_params,
            fast_windows=fast_windows,
            slow_windows=slow_windows,
            initial_capital=initial_capital,
            tmp_dir=tmp_dir,
            fold_idx=idx,
            commission_pct=commission_pct,
            slippage_bps=slippage_bps,
            risk_fraction=risk_fraction,
            trading_days=trading_days,
            buffer_size=buffer_size,
            regime_filter_cfg=regime_filter_cfg,
        )
        print(f"best {param_a}={best_fast}, {param_b}={best_slow}, "
              f"IS Sharpe={is_sharpe:.3f}")

        print(f"    Running OOS backtest...", end=" ", flush=True)
        oos_db = os.path.join(tmp_dir, f"oos_final_{idx}.db")
        params = {**base_params, param_a: best_fast, param_b: best_slow}
        oos_sharpe, oos_equity = run_silent_wf(
            data_dir=oos_data_dir,
            rows=oos_rows_primary,
            tickers=tickers,
            aux_tickers=aux_tickers,
            strategy_type=strategy_type,
            strategy_params=params,
            initial_capital=initial_capital,
            risk_fraction=risk_fraction,
            commission_pct=commission_pct,
            slippage_bps=slippage_bps,
            trading_days=trading_days,
            buffer_size=buffer_size,
            db_path=oos_db,
            regime_filter_cfg=regime_filter_cfg,
        )
        print(f"OOS Sharpe={oos_sharpe:.3f}")

        results.append({
            "fold":       idx + 1,
            "is_start":   fold["is_start"],
            "is_end":     fold["is_end"],
            "oos_start":  fold["oos_start"],
            "oos_end":    fold["oos_end"],
            "best_fast":  best_fast,
            "best_slow":  best_slow,
            "is_sharpe":  round(is_sharpe,  4),
            "oos_sharpe": round(oos_sharpe, 4) if not np.isnan(oos_sharpe) else None,
            "oos_equity": oos_equity,
            "is_bars":    len(is_rows_primary),
            "oos_bars":   len(oos_rows_primary),
        })

    return results


# ---------------------------------------------------------------
# Stitched OOS equity
# ---------------------------------------------------------------

def stitch_oos_equity(results: List[Dict], initial_capital: float) -> np.ndarray:
    stitched = []
    carry = initial_capital
    for res in results:
        eq = res["oos_equity"]
        if len(eq) == 0:
            continue
        scaled = eq / eq[0] * carry
        stitched.extend(scaled.tolist())
        carry = scaled[-1]
    return np.array(stitched, dtype=float)


# ---------------------------------------------------------------
# Metrics — trading_days passed explicitly
# ---------------------------------------------------------------

def compute_wf_metrics(equities: np.ndarray, trading_days: int) -> Dict:
    """
    Named compute_wf_metrics (not compute_metrics) to avoid shadowing
    the function of the same name in engine_analytics.py if both are
    imported in the same session.
    """
    if len(equities) < 2:
        return {}
    rets   = np.diff(equities) / equities[:-1]
    std    = rets.std()
    sharpe = float(rets.mean() / std * np.sqrt(trading_days)) if std > 1e-10 else 0.0
    peak   = np.maximum.accumulate(equities)
    dd     = (equities - peak) / peak
    max_dd = float(dd.min())
    ann_ret = float(
        (equities[-1] / equities[0]) ** (trading_days / len(equities)) - 1
    )
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-10 else float("inf")
    return {
        "sharpe":   round(sharpe, 4),
        "max_dd":   round(max_dd * 100, 4),
        "ann_ret":  round(ann_ret * 100, 4),
        "calmar":   round(calmar, 4),
        "final_eq": round(float(equities[-1]), 2),
    }


# ---------------------------------------------------------------
# Save results CSV
# ---------------------------------------------------------------

def save_results_csv(results: List[Dict], out_dir: Path) -> None:
    path = out_dir / "walk_forward_results.csv"
    with open(path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=[
            "fold", "is_start", "is_end", "oos_start", "oos_end",
            "best_fast", "best_slow", "is_sharpe", "oos_sharpe",
            "is_bars", "oos_bars",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: v for k, v in r.items() if k != "oos_equity"})
    print(f"[wf] Results CSV → {path}")


# ---------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------

def plot_walk_forward(
    results: List[Dict],
    stitched: np.ndarray,
    initial_capital: float,
    trading_days: int,
    out_dir: Path,
    tickers_str: str,
    strategy_name: str,
    method: str,
) -> None:
    DARK_BG  = "#0f0f0f"
    PANEL_BG = "#1a1a1a"
    BLUE     = "#4fa3e0"
    GREEN    = "#52e07a"
    RED      = "#e05252"
    ORANGE   = "#f0a500"
    TEXT     = "#d0d0d0"
    GRID_C   = "#2a2a2a"

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(DARK_BG)
    gs = GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35,
                  height_ratios=[1.8, 1.8, 1.4])

    ax_timeline = fig.add_subplot(gs[0, :])
    ax_equity   = fig.add_subplot(gs[1, :])
    ax_sharpe   = fig.add_subplot(gs[2, 0])
    ax_params   = fig.add_subplot(gs[2, 1])

    def _style(ax):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_C)
        ax.grid(color=GRID_C, lw=0.5, ls="--", alpha=0.6)

    method_label = method.capitalize()

    # Panel 1: Fold timeline
    _style(ax_timeline)
    ax_timeline.set_title(
        f"Walk-Forward Fold Structure  [{method_label}]", fontsize=11, pad=8
    )
    ax_timeline.set_yticks(range(len(results)))
    ax_timeline.set_yticklabels(
        [f"Fold {r['fold']}" for r in results], fontsize=8, color=TEXT
    )
    ax_timeline.set_xlabel("Bar index proxy", fontsize=9)

    for i, res in enumerate(results):
        is_len  = res["is_bars"]
        oos_len = res["oos_bars"]
        offset  = (
            sum(r["is_bars"] for r in results[:i])
            if method == "anchored"
            else i * (is_len + oos_len)
        )
        ax_timeline.barh(i, is_len,  left=offset,
                         color=BLUE,   alpha=0.55, height=0.5,
                         label="IS"  if i == 0 else "")
        ax_timeline.barh(i, oos_len, left=offset + is_len,
                         color=ORANGE, alpha=0.8,  height=0.5,
                         label="OOS" if i == 0 else "")
        ax_timeline.text(
            offset + is_len + oos_len / 2, i,
            f"a={res['best_fast']}/b={res['best_slow']}",
            ha="center", va="center", fontsize=6.5,
            color="black", fontweight="bold",
        )
    ax_timeline.legend(facecolor=PANEL_BG, edgecolor=GRID_C,
                       labelcolor=TEXT, fontsize=8, loc="lower right")

    # Panel 2: Stitched OOS equity
    _style(ax_equity)
    ax_equity.plot(range(len(stitched)), stitched, color=GREEN, lw=1.5,
                   label="Stitched OOS Equity")
    ax_equity.axhline(initial_capital, color=TEXT, ls="--", lw=0.7,
                      alpha=0.4, label="Initial Capital")

    boundary = 0
    for i, res in enumerate(results):
        if i > 0:
            ax_equity.axvline(boundary, color=ORANGE, lw=0.8, ls=":",
                              alpha=0.6, label="Fold boundary" if i == 1 else "")
        boundary += len(res["oos_equity"])

    oos_m = compute_wf_metrics(stitched, trading_days)
    subtitle = (
        f"Sharpe: {oos_m.get('sharpe','—')}   "
        f"Max DD: {oos_m.get('max_dd','—')}%   "
        f"Ann. Return: {oos_m.get('ann_ret','—')}%   "
        f"Calmar: {oos_m.get('calmar','—')}"
    )
    ax_equity.set_title(
        f"Stitched Out-of-Sample Equity Curve\n{subtitle}", fontsize=10, pad=8
    )
    ax_equity.set_ylabel("Portfolio Value ($)", fontsize=9)
    ax_equity.set_xlabel("OOS Bar Index", fontsize=9)
    ax_equity.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
    )
    ax_equity.legend(facecolor=PANEL_BG, edgecolor=GRID_C,
                     labelcolor=TEXT, fontsize=7)

    # Panel 3: IS vs OOS Sharpe per fold
    _style(ax_sharpe)
    folds_x     = [r["fold"] for r in results]
    is_sharpes  = [r["is_sharpe"] for r in results]
    oos_sharpes = [r["oos_sharpe"] if r["oos_sharpe"] is not None else 0.0
                   for r in results]
    w  = 0.35
    xs = np.arange(len(folds_x))
    ax_sharpe.bar(xs - w/2, is_sharpes,  width=w, color=BLUE,   alpha=0.8, label="IS Sharpe")
    ax_sharpe.bar(xs + w/2, oos_sharpes, width=w, color=ORANGE, alpha=0.8, label="OOS Sharpe")
    ax_sharpe.axhline(0, color=TEXT,  lw=0.6, alpha=0.4)
    ax_sharpe.axhline(1, color=GREEN, lw=0.8, ls=":", alpha=0.5, label="Sharpe = 1")
    ax_sharpe.set_xticks(xs)
    ax_sharpe.set_xticklabels([f"F{f}" for f in folds_x], fontsize=8, color=TEXT)
    ax_sharpe.set_title(
        "IS vs OOS Sharpe by Fold\n(large IS→OOS gap = overfitting)", fontsize=9, pad=8
    )
    ax_sharpe.set_ylabel("Annualised Sharpe", fontsize=9)
    ax_sharpe.legend(facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor=TEXT, fontsize=7)

    # Panel 4: Best parameters per fold
    _style(ax_params)
    fasts = [r["best_fast"] for r in results]
    slows = [r["best_slow"] for r in results]
    ax_params.plot(folds_x, fasts, "o--", color=BLUE,   lw=1.2, ms=6, label="Best param_a")
    ax_params.plot(folds_x, slows, "s--", color=ORANGE, lw=1.2, ms=6, label="Best param_b")
    ax_params.set_xticks(folds_x)
    ax_params.set_xticklabels([f"Fold {f}" for f in folds_x], fontsize=8, color=TEXT)
    ax_params.set_title(
        "IS-Optimised Parameters by Fold\n(stable = robust; erratic = unstable)",
        fontsize=9, pad=8,
    )
    ax_params.set_ylabel("Window (bars)", fontsize=9)
    ax_params.legend(facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor=TEXT, fontsize=7)

    fig.text(
        0.5, 0.98,
        f"Walk-Forward Validation [{method_label}]  ·  {tickers_str}  ·  {strategy_name}",
        ha="center", va="top",
        fontsize=14, color="white", fontweight="bold",
    )

    save_path = out_dir / "walk_forward.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[wf] Chart saved → {save_path}")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def main() -> None:
    # Step 1: resolve --run before reading config
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--run", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    out_dir = Path(pre_args.run) if pre_args.run else find_latest_output_dir()

    cfg = load_run_config(out_dir)
    if not cfg:
        print(f"[error] No config_snapshot.yaml found in {out_dir}")
        sys.exit(1)

    # ── Extract all config values — no hardcoded fallbacks ────────────────
    cfg_data      = require(cfg, "data",        "root")
    cfg_strategy  = require(cfg, "strategy",    "root")
    cfg_risk      = require(cfg, "risk",         "root")
    cfg_exec      = require(cfg, "execution",    "root")
    cfg_analytics = require(cfg, "analytics",    "root")
    cfg_filter = cfg.get("regime_filter")

    strategy_type   = require(cfg_strategy,  "type",                   "strategy")
    # Strip "type" before any downstream use — _build_strategy() must not receive it
    base_params     = {k: v for k, v in cfg_strategy.items() if k != "type"}

    # Backward-compat: support old `ticker: "SPY"` single-string format
    tickers = cfg_data.get("tickers") or cfg_data.get("ticker")
    if tickers is None:
        print("[error] Missing required config key: data.tickers")
        sys.exit(1)
    if isinstance(tickers, str):
        tickers = [tickers]
    aux_tickers: List[str] = cfg_data.get("aux_tickers", [])

    initial_capital = float(require(cfg_risk,      "initial_capital",       "risk"))
    risk_fraction   = float(require(cfg_risk,      "risk_fraction",         "risk"))
    commission_pct  = float(require(cfg_exec,      "commission_pct",        "execution"))
    slippage_bps    = float(require(cfg_exec,      "slippage_bps",          "execution"))
    trading_days    = int(require(cfg_analytics,   "trading_days_per_year", "analytics"))
    buffer_size     = int(require(cfg_data,        "buffer_size",           "data"))
    start           = require(cfg_data,            "start",                 "data")
    end             = require(cfg_data,            "end",                   "data")
    data_dir        = cfg_data.get("data_dir", "data")

    wf_cfg = cfg.get("walk_forward")
    if not wf_cfg:
        print("[error] Missing 'walk_forward' block in config. "
              "Add method, oos_months, is_months, and folds.")
        sys.exit(1)

    method     = require(wf_cfg, "method",     "walk_forward")
    oos_months = int(require(wf_cfg, "oos_months", "walk_forward"))
    is_months  = int(require(wf_cfg, "is_months",  "walk_forward"))
    n_folds    = int(require(wf_cfg, "folds",       "walk_forward"))

    if method not in ("anchored", "rolling"):
        print(f"[error] walk_forward.method must be 'anchored' or 'rolling', got '{method}'")
        sys.exit(1)

    default_fast, default_slow = get_is_windows(strategy_type)

    # Step 2: full parser — CLI can override config values
    parser = argparse.ArgumentParser(description="Walk-forward validation.")
    parser.add_argument("--run",        type=str,  default=None)
    parser.add_argument("--folds",      type=int,  default=n_folds)
    parser.add_argument("--oos_months", type=int,  default=oos_months)
    parser.add_argument("--is_months",  type=int,  default=is_months)
    parser.add_argument("--method",     type=str,  default=method,
                        choices=["anchored", "rolling"])
    parser.add_argument("--fast",       type=int,  nargs="+", default=default_fast)
    parser.add_argument("--slow",       type=int,  nargs="+", default=default_slow)
    args = parser.parse_args()

    tickers_str = ", ".join(tickers)

    # Ensure all ticker data is cached before the fold loop starts
    for t in tickers + aux_tickers:
        fetch_data(ticker=t, start=start, end=end, data_dir=Path(data_dir))

    csv_path = Path(data_dir) / f"{tickers[0]}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Primary ticker CSV not found: {csv_path}. "
            "fetch_data() should have created it above."
        )

    print(f"\n{'='*54}")
    print(f"  Walk-Forward Validation — {tickers_str} {strategy_type}")
    print(f"  Method       : {args.method.upper()}")
    print(f"  Folds        : {args.folds}")
    print(f"  OOS window   : {args.oos_months} months per fold")
    print(f"  IS window    : {args.is_months} months"
          f"  {'(fixed)' if args.method == 'rolling' else '(ignored — anchored uses full history)'}")
    print(f"  IS grid      : {args.fast}  /  {args.slow}")
    print(f"  Capital      : ${initial_capital:,.0f}")
    print(f"  Risk fraction: {risk_fraction}")
    print(f"  Trading days : {trading_days}")
    print(f"  Output dir   : {out_dir}")
    print(f"{'='*54}\n")

    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_walk_forward(
            csv_path=csv_path,
            tickers=tickers,
            aux_tickers=aux_tickers,
            strategy_type=strategy_type,
            base_params=base_params,
            fast_windows=args.fast,
            slow_windows=args.slow,
            initial_capital=initial_capital,
            n_folds=args.folds,
            oos_months=args.oos_months,
            method=args.method,
            is_months=args.is_months,
            commission_pct=commission_pct,
            slippage_bps=slippage_bps,
            risk_fraction=risk_fraction,
            trading_days=trading_days,
            buffer_size=buffer_size,
            tmp_dir=tmp_dir,
            regime_filter_cfg=cfg_filter,
        )

    print(f"\n{'─'*46}")
    print("  Summary")
    print(f"{'─'*46}")
    for r in results:
        oos_s = f"{r['oos_sharpe']:.3f}" if r["oos_sharpe"] is not None else " n/a"
        print(
            f"  Fold {r['fold']}  IS Sharpe={r['is_sharpe']:>6.3f}  "
            f"OOS Sharpe={oos_s:>6}  "
            f"params=({r['best_fast']},{r['best_slow']})"
        )

    stitched = stitch_oos_equity(results, initial_capital)
    oos_m    = compute_wf_metrics(stitched, trading_days)

    print(f"\n  Combined OOS metrics ({args.method}):")
    print(f"    Sharpe     : {oos_m.get('sharpe')}")
    print(f"    Max DD     : {oos_m.get('max_dd')}%")
    print(f"    Ann Return : {oos_m.get('ann_ret')}%")
    print(f"    Calmar     : {oos_m.get('calmar')}")

    save_results_csv(results, out_dir)
    plot_walk_forward(
        results, stitched, initial_capital, trading_days,
        out_dir, tickers_str, strategy_type, method=args.method,
    )


if __name__ == "__main__":
    main()
