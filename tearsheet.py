# =============================================================
# tearsheet.py
# Generates a professional multi-panel strategy tearsheet.
#
# Panels:
#   [1] Equity curve vs. Buy-and-Hold benchmark
#   [2] Drawdown
#   [3] Rolling 60-bar Sharpe ratio
#   [4] Monthly returns heatmap
#
# Usage:
#   python tearsheet.py
#   python tearsheet.py --run outputs/ma_cross_SPY_20260320_190000
# =============================================================
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# Import shared helpers from visualize.py — avoids duplication of
# find_latest_output_dir() which must sort by st_mtime, not alphabetically
from visualize import find_latest_output_dir


# ---------------------------------------------------------------
# Config reader
# ---------------------------------------------------------------

def _read_config(out_dir: Path) -> Tuple[str, str, float, int]:
    """
    Extract (tickers_str, strategy_name, initial_capital, trading_days)
    from config_snapshot.yaml.

    tickers_str: comma-joined list of tradeable tickers — used in title
                 and to select the B&H benchmark (tickers[0]).
    Returns generic labels if the snapshot is missing.
    Never hardcodes a ticker name or capital amount as a fallback.
    """
    config_path = out_dir / "config_snapshot.yaml"
    if not config_path.exists():
        return "Unknown", "Strategy", 100_000.0, 252

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    strategy_name: str = cfg.get("strategy", {}).get("type", "Strategy")

    cfg_data = cfg.get("data", {})
    tickers = cfg_data.get("tickers") or cfg_data.get("ticker")
    if tickers is None:
        tickers_str = "Unknown"
    elif isinstance(tickers, list):
        tickers_str = ", ".join(tickers)
    else:
        tickers_str = str(tickers)

    # initial_capital must come from config — no hardcoded fallback
    cfg_risk = cfg.get("risk", {})
    initial_capital: float = float(cfg_risk.get("initial_capital", 100_000.0))

    trading_days: int = int(
        cfg.get("analytics", {}).get("trading_days_per_year", 252)
    )

    return tickers_str, strategy_name, initial_capital, trading_days


def _primary_ticker(out_dir: Path) -> str:
    """Return the first tradeable ticker for the B&H benchmark."""
    config_path = out_dir / "config_snapshot.yaml"
    if not config_path.exists():
        return "SPY"
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg_data = cfg.get("data", {})
    tickers = cfg_data.get("tickers") or cfg_data.get("ticker")
    if isinstance(tickers, list):
        return tickers[0]
    if isinstance(tickers, str):
        return tickers
    return "SPY"


# ---------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------

def load_equity_curve(db_path: Path) -> Tuple[List[str], np.ndarray]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute(
        "SELECT timestamp, equity FROM equity_curve ORDER BY id"
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        raise RuntimeError("equity_curve table is empty.")
    timestamps = [r[0] for r in rows]
    equities = np.array([r[1] for r in rows], dtype=float)
    return timestamps, equities


def load_metrics(out_dir: Path) -> Dict[str, str]:
    metrics_path = out_dir / "metrics.csv"
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        lines = f.read().strip().splitlines()
    if len(lines) >= 2:
        headers = lines[0].split(",")
        values = lines[1].split(",")
        return dict(zip(headers, values))
    return {}


# ---------------------------------------------------------------
# Computations
# ---------------------------------------------------------------

def rolling_sharpe(
    returns: np.ndarray,
    window: int,
    trading_days: int,
) -> np.ndarray:
    rs = np.full(len(returns), np.nan)
    for i in range(window, len(returns)):
        w = returns[i - window: i]
        std = w.std()
        rs[i] = float(w.mean() / std * np.sqrt(trading_days)) if std > 1e-10 else 0.0
    return rs


def build_monthly_returns(
    timestamps: List[str], equities: np.ndarray
) -> Tuple[List[int], List[int], np.ndarray]:
    monthly_last: Dict[Tuple[int, int], float] = {}
    for ts, eq in zip(timestamps, equities):
        date_part = str(ts)[:10]
        year, month = int(date_part[:4]), int(date_part[5:7])
        monthly_last[(year, month)] = eq

    sorted_keys = sorted(monthly_last.keys())
    if len(sorted_keys) < 2:
        return [], [], np.array([])

    years = sorted(set(k[0] for k in sorted_keys))
    matrix = np.full((len(years), 12), np.nan)
    prev_eq: Optional[float] = None

    for year, month in sorted_keys:
        curr_eq = monthly_last[(year, month)]
        if prev_eq is not None and prev_eq > 0:
            ret = (curr_eq / prev_eq - 1.0) * 100.0
            yi = years.index(year)
            matrix[yi][month - 1] = round(ret, 2)
        prev_eq = curr_eq

    return years, list(range(1, 13)), matrix


# ---------------------------------------------------------------
# Buy-and-Hold benchmark
# ---------------------------------------------------------------

def build_bah_curve(
    ticker: str,
    timestamps: List[str],
    initial_capital: float,
) -> Optional[np.ndarray]:
    try:
        import warnings
        import yfinance as yf
        warnings.filterwarnings("ignore")

        start = str(timestamps[0])[:10]
        end = str(timestamps[-1])[:10]

        raw = yf.download(
            ticker, start=start, end=end,
            auto_adjust=True, multi_level_index=False, progress=False,
        )
        if raw.empty:
            return None

        raw = raw.reset_index()
        if hasattr(raw.columns, "get_level_values"):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]

        # Normalise date column — yfinance daily uses "date", intraday "datetime"
        for alias in ("datetime", "timestamp"):
            if alias in raw.columns:
                raw = raw.rename(columns={alias: "date"})

        close_map = {
            str(row["date"])[:10]: float(row["close"])
            for _, row in raw.iterrows()
        }

        prices: List[float] = []
        for ts in timestamps:
            key = str(ts)[:10]
            if key in close_map:
                prices.append(close_map[key])
            elif prices:
                prices.append(prices[-1])

        if not prices:
            return None

        bah = np.array(prices, dtype=float)
        bah = bah / bah[0] * initial_capital
        return bah[:len(timestamps)]

    except Exception:
        return None


# ---------------------------------------------------------------
# Core plotting
# ---------------------------------------------------------------

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

DARK_BG  = "#0f0f0f"
PANEL_BG = "#1a1a1a"
BLUE     = "#4fa3e0"
ORANGE   = "#f0a500"
RED      = "#e05252"
GREEN    = "#52e07a"
TEXT     = "#d0d0d0"
GRID     = "#2a2a2a"


def _style_ax(ax) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.7)


def _set_date_xticks(ax, timestamps: List[str], n_ticks: int = 8) -> None:
    """Replace bar-index x-axis with real date labels, thinned to n_ticks."""
    n = len(timestamps)
    if n < 2:
        return
    tick_locs = [int(i * (n - 1) / (n_ticks - 1)) for i in range(n_ticks)]
    tick_labels = [timestamps[i][:10] for i in tick_locs]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7)


def build_tearsheet(
    out_dir: Path,
    ticker_override: Optional[str] = None,     # None → use config
    capital_override: Optional[float] = None,  # None → use config
    rolling_window: int = 60,
) -> None:
    tickers_str, strategy_name, initial_capital, trading_days = _read_config(out_dir)

    # CLI overrides only apply when explicitly passed — config always wins otherwise
    bah_ticker: str = ticker_override or _primary_ticker(out_dir)
    if capital_override is not None:
        initial_capital = capital_override

    db_path = out_dir / "backtest.db"
    if not db_path.exists():
        print(f"[error] backtest.db not found in {out_dir}")
        sys.exit(1)

    timestamps, equities = load_equity_curve(db_path)
    metrics = load_metrics(out_dir)

    returns = np.diff(equities) / equities[:-1]
    peak = np.maximum.accumulate(equities)
    drawdown = (equities - peak) / peak * 100
    rs = rolling_sharpe(returns, window=rolling_window, trading_days=trading_days)
    years, _, monthly_matrix = build_monthly_returns(timestamps, equities)

    print(f"[tearsheet] Fetching {bah_ticker} buy-and-hold benchmark...")
    bah = build_bah_curve(bah_ticker, timestamps, initial_capital)

    n = len(equities)
    x = range(n)

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(DARK_BG)

    gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=[2.5, 1.5, 2.5],
        hspace=0.45, wspace=0.3,
    )
    ax_equity  = fig.add_subplot(gs[0, :])
    ax_dd      = fig.add_subplot(gs[1, :])
    ax_rs      = fig.add_subplot(gs[2, 0])
    ax_heatmap = fig.add_subplot(gs[2, 1])

    # Panel 1: Equity curve
    _style_ax(ax_equity)
    ax_equity.plot(x, equities, color=BLUE, lw=1.5, label="Strategy")
    ax_equity.axhline(
        equities[0], color=TEXT, ls="--", lw=0.7, alpha=0.4,
        label="Initial Capital",
    )
    if bah is not None:
        ax_equity.plot(
            x[:len(bah)], bah, color=ORANGE, lw=1.2, ls="--",
            alpha=0.8, label=f"{bah_ticker} Buy & Hold",
        )
    ax_equity.set_title("Equity Curve vs. Buy & Hold Benchmark", fontsize=12, pad=10)
    ax_equity.set_ylabel("Portfolio Value ($)", fontsize=9)
    ax_equity.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
    )
    ax_equity.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    _set_date_xticks(ax_equity, timestamps)

    # Panel 2: Drawdown
    _style_ax(ax_dd)
    ax_dd.fill_between(x, drawdown, 0, color=RED, alpha=0.5)
    ax_dd.plot(x, drawdown, color=RED, lw=0.8, alpha=0.7)
    ax_dd.set_title("Drawdown (%)", fontsize=10, pad=8)
    ax_dd.set_ylabel("Drawdown (%)", fontsize=9)
    _set_date_xticks(ax_dd, timestamps)
    ax_dd.set_xlabel("Date", fontsize=9)

    # Panel 3: Rolling Sharpe
    _style_ax(ax_rs)
    rs_x = range(len(rs))
    ax_rs.plot(rs_x, rs, color=GREEN, lw=1.0, alpha=0.9)
    ax_rs.axhline(0,     color=TEXT,   ls="--", lw=0.6, alpha=0.4)
    ax_rs.axhline(1,     color=ORANGE, ls=":",  lw=0.8, alpha=0.5, label="Sharpe = 1")
    ax_rs.fill_between(rs_x, rs, 0, where=np.nan_to_num(rs) >= 0,
                       color=GREEN, alpha=0.15)
    ax_rs.fill_between(rs_x, rs, 0, where=np.nan_to_num(rs) < 0,
                       color=RED, alpha=0.15)
    ax_rs.set_title(f"Rolling {rolling_window}-Bar Sharpe", fontsize=10, pad=8)
    ax_rs.set_ylabel("Sharpe Ratio", fontsize=9)
    _set_date_xticks(ax_rs, timestamps)
    ax_rs.set_xlabel("Date", fontsize=9)
    ax_rs.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=7)

    # Panel 4: Monthly Returns Heatmap
    ax_heatmap.set_facecolor(PANEL_BG)
    ax_heatmap.title.set_color(TEXT)
    ax_heatmap.tick_params(colors=TEXT, labelsize=7)
    for spine in ax_heatmap.spines.values():
        spine.set_edgecolor(GRID)

    if monthly_matrix.size > 0:
        abs_max = float(np.nanmax(np.abs(monthly_matrix)))
        norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        im = ax_heatmap.imshow(
            monthly_matrix, cmap="RdYlGn", norm=norm, aspect="auto"
        )
        ax_heatmap.set_xticks(range(12))
        ax_heatmap.set_xticklabels(MONTH_LABELS, fontsize=7, color=TEXT)
        ax_heatmap.set_yticks(range(len(years)))
        ax_heatmap.set_yticklabels(years, fontsize=7, color=TEXT)
        for i in range(len(years)):
            for j in range(12):
                val = monthly_matrix[i, j]
                if not np.isnan(val):
                    ax_heatmap.text(
                        j, i, f"{val:.1f}%",
                        ha="center", va="center", fontsize=5.5,
                        color="black" if abs(val) < abs_max * 0.6 else "white",
                    )
        cbar = fig.colorbar(im, ax=ax_heatmap, fraction=0.03, pad=0.04)
        cbar.ax.tick_params(colors=TEXT, labelsize=7)
        cbar.ax.yaxis.label.set_color(TEXT)

    ax_heatmap.set_title("Monthly Returns Heatmap (%)", fontsize=10, pad=8)
    ax_heatmap.set_xlabel("Month", fontsize=9)
    ax_heatmap.set_ylabel("Year", fontsize=9)

    # Metrics banner — key names must match compute_metrics() output exactly
    banner_items = [
        ("Sharpe",       metrics.get("sharpe_ratio",    "—")),
        ("Max DD",       f"{metrics.get('max_drawdown_pct', '—')}%"),
        ("Calmar",       metrics.get("calmar_ratio",    "—")),
        ("Hit Rate",     f"{metrics.get('hit_rate_pct', '—')}%"),
        ("Ann. Return",  f"{metrics.get('ann_return_pct', '—')}%"),
        ("Turnover",     metrics.get("turnover",        "—")),   # FIXED: was "portfolio_turnover"
        ("Trades",       metrics.get("total_trades",    "—")),
        ("Final Equity", f"${float(metrics['final_equity']):,.0f}"
                         if metrics.get("final_equity") else "—"),
    ]
    banner_text = "     |     ".join(f"{k}: {v}" for k, v in banner_items)
    fig.text(
        0.5, 0.975, banner_text,
        ha="center", va="top", fontsize=8.5, color=TEXT,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL_BG,
                  edgecolor=GRID, alpha=0.9),
    )
    fig.text(
        0.5, 1.005,
        f"Strategy Tearsheet  ·  {tickers_str}  ·  {strategy_name}",
        ha="center", va="bottom",
        fontsize=14, color="white", fontweight="bold",
    )

    save_path = out_dir / "tearsheet.png"
    plt.savefig(
        str(save_path), dpi=150, bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close()
    print(f"[tearsheet] Saved → {save_path}")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strategy tearsheet.")
    parser.add_argument("--run",     type=str,   default=None)
    parser.add_argument("--ticker",  type=str,   default=None,
                        help="Override B&H benchmark ticker (default: from config).")
    parser.add_argument("--capital", type=float, default=None,
                        help="Override initial capital (default: from config).")
    parser.add_argument("--rolling", type=int,   default=60)
    args = parser.parse_args()

    out_dir = Path(args.run) if args.run else find_latest_output_dir()
    print(f"[tearsheet] Run folder: {out_dir}")

    build_tearsheet(
        out_dir=out_dir,
        ticker_override=args.ticker,       # None → config wins
        capital_override=args.capital,     # None → config wins
        rolling_window=args.rolling,
    )


if __name__ == "__main__":
    main()
