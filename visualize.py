# =============================================================
# visualize.py
# Auto-detects the latest outputs/ run and plots equity curve
# + drawdown from its backtest.db.
#
# Usage:
#   python visualize.py                          # latest run
#   python visualize.py --run outputs/my_run/   # specific run
# =============================================================
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def find_latest_output_dir() -> Path:
    outputs = Path("outputs")
    if not outputs.exists():
        raise FileNotFoundError("No outputs/ directory found. Run the backtest first.")
    runs = [d for d in outputs.iterdir() if d.is_dir()]
    if not runs:
        raise FileNotFoundError("No run folders found in outputs/. Run the backtest first.")
    # Sort by modification time — alphabetical sort breaks when run names differ
    return max(runs, key=lambda d: d.stat().st_mtime)


def _read_config_meta(out_dir: Path):
    """
    Extract (tickers_str, strategy_name) from config_snapshot.yaml.
    Returns generic labels if the snapshot is missing — never hardcodes
    a ticker name as a fallback.
    """
    config_path = out_dir / "config_snapshot.yaml"
    if not config_path.exists():
        return "Unknown", "Strategy"

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    strategy_name: str = cfg.get("strategy", {}).get("type", "Strategy")

    cfg_data = cfg.get("data", {})
    # Support both old `ticker: "SPY"` and new `tickers: ["SPY"]` formats
    tickers = cfg_data.get("tickers") or cfg_data.get("ticker")
    if tickers is None:
        tickers_str = "Unknown"
    elif isinstance(tickers, list):
        tickers_str = ", ".join(tickers)
    else:
        tickers_str = str(tickers)

    return tickers_str, strategy_name


def plot(db_path: Path, out_dir: Path) -> None:
    tickers_str, strategy_name = _read_config_meta(out_dir)

    conn = sqlite3.connect(str(db_path))
    cur = conn.execute(
        "SELECT timestamp, equity FROM equity_curve ORDER BY id"
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("[error] equity_curve table is empty.")
        sys.exit(1)

    timestamps: List[str] = [r[0] for r in rows]
    equities: List[float] = [r[1] for r in rows]

    eq_arr: np.ndarray = np.array(equities, dtype=float)
    peak: np.ndarray = np.maximum.accumulate(eq_arr)
    dd: np.ndarray = (eq_arr - peak) / peak * 100

    n = len(eq_arr)
    x = range(n)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ── Equity curve ──────────────────────────────────────────────────────
    ax1.plot(x, eq_arr, color="#1f77b4", lw=1.5)
    ax1.axhline(eq_arr[0], color="grey", ls="--", lw=0.8, label="Initial Capital")
    ax1.set_title(
        f"{tickers_str} — {strategy_name} Backtest | Equity Curve", fontsize=13
    )
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda val, _: f"${val:,.0f}"
    ))
    ax1.legend(fontsize=9)

    # ── Drawdown ──────────────────────────────────────────────────────────
    ax2.fill_between(x, dd, 0, color="red", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")

    # ── X-axis: use real timestamps, thinned to ~10 labels ────────────────
    # Bar-index x-axis is unreadable for multi-year backtests — use dates.
    n_ticks = min(10, n)
    tick_locs = [int(i * (n - 1) / (n_ticks - 1)) for i in range(n_ticks)]
    tick_labels = [timestamps[i][:10] for i in tick_locs]   # "YYYY-MM-DD"
    ax2.set_xticks(tick_locs)
    ax2.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    ax2.set_xlabel("Date")

    plt.tight_layout()

    save_path = out_dir / "equity_curve.png"
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"[visualize] Chart saved → {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot equity curve and drawdown from a backtest run."
    )
    parser.add_argument(
        "--run", type=str, default=None,
        help="Path to a specific run folder. Defaults to the most recent run.",
    )
    args = parser.parse_args()

    out_dir = Path(args.run) if args.run else find_latest_output_dir()
    db_path = out_dir / "backtest.db"

    if not db_path.exists():
        print(f"[error] backtest.db not found in {out_dir}")
        sys.exit(1)

    print(f"[visualize] Reading from: {db_path}")
    plot(db_path, out_dir)


if __name__ == "__main__":
    main()
