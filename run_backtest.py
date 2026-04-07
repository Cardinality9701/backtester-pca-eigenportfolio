# =============================================================
# run_backtest.py
# Config-driven CLI entry point.
#
# All component wiring is handled inside run_backtest() in
# src/engine_analytics.py. This file only handles:
#   1. Config parsing and validation
#   2. Data fetching / caching
#   3. Output directory creation and artifact saving
#   4. Calling run_backtest() with the flat parameter signature
#
# Usage:
#   python run_backtest.py --config config/tsmom_SPY.yaml
#   python run_backtest.py --config config/bollinger_SPY_vix.yaml --refresh
# =============================================================
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.engine_analytics import run_backtest


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def require(cfg_dict: Dict, key: str, section: str) -> Any:
    """Exit loudly if a required config key is missing. No silent defaults."""
    val = cfg_dict.get(key)
    if val is None:
        print(f"[error] Missing required config key: {section}.{key}")
        sys.exit(1)
    return val


# Tickers whose yfinance download symbol differs from their clean name.
# Add new entries here as needed — no other file needs to change.
_YFINANCE_SYMBOL_MAP: Dict[str, str] = {
    "VIX":  "^VIX",
    "VIX3M": "^VIX3M", # VIX 3 month
    "GSPC": "^GSPC",   # S&P 500 Price Index
    "DJI":  "^DJI",    # Dow Jones Index
    "TNX":  "^TNX",    # 10-yr Treasury yield
    "TYX":  "^TYX",    # 30-yr Treasury yield
}


def fetch_data(
    ticker: str,
    start: str,
    end: str,
    data_dir: Path,
    refresh: bool = False,
) -> None:
    """
    Download OHLCV data for `ticker` via yfinance and cache to CSV.

    ticker   : clean name used throughout the codebase (e.g. "VIX", "SPY")
    The yfinance download symbol may differ (e.g. "^VIX") — resolved
    internally via _YFINANCE_SYMBOL_MAP. The CSV is always saved under
    the clean ticker name so all downstream code is unaffected.

    Column naming contract: saves as 'date' so that
    CSVParquetDataHandler._load_file() can find pl.col("date").
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{ticker}.csv"

    if csv_path.exists() and not refresh:
        print(f"[data] Using cached data: {csv_path}")
        return

    # Resolve yfinance symbol — defaults to the clean name if not in map
    yf_symbol = _YFINANCE_SYMBOL_MAP.get(ticker.upper(), ticker)
    print(f"[data] Downloading {ticker} (yf: {yf_symbol}) ({start} → {end})...")

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required. Run: pip install yfinance")

    raw = yf.download(
        yf_symbol, start=start, end=end,
        auto_adjust=True, multi_level_index=False, progress=False,
    )
    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for '{yf_symbol}' (ticker='{ticker}'). "
            f"Check the symbol map or date range."
        )

    raw = raw.reset_index()
    if hasattr(raw.columns, "get_level_values"):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]

    # Normalise date column → "date"
    for alias in ("datetime", "timestamp"):
        if alias in raw.columns:
            raw = raw.rename(columns={alias: "date"})

    required_cols = ["date", "open", "high", "low", "close", "volume"]
    raw = raw[[c for c in required_cols if c in raw.columns]]

    # VIX has no volume — fill with 0 so downstream schema stays consistent
    if "volume" not in raw.columns:
        raw["volume"] = 0

    raw.to_csv(csv_path, index=False)
    print(f"[data] Saved {len(raw):,} rows → {csv_path}")


def create_output_dir(run_name: str, primary_ticker: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / f"{run_name}_{primary_ticker}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_config_snapshot(cfg: Dict, out_dir: Path) -> None:
    path = out_dir / "config_snapshot.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"[artifacts] Config snapshot → {path}")


def save_metrics_csv(metrics_df, out_dir: Path) -> None:
    path = out_dir / "metrics.csv"
    metrics_df.write_csv(path)
    print(f"[artifacts] Metrics → {path}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven backtesting engine.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--refresh", action="store_true", default=False,
        help="Re-download data even if a cached CSV exists.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[error] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg: Dict = yaml.safe_load(f)

    run_name: str = cfg.get("run_name", "backtest")
    print(f"\n{'='*54}")
    print(f"  Run    : {run_name}")
    print(f"  Config : {config_path}")
    print(f"{'='*54}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    cfg_data = require(cfg, "data", "root")

    # Backward-compat: support old `ticker: "SPY"` single-string format
    tickers = cfg_data.get("tickers") or cfg_data.get("ticker")
    if tickers is None:
        print("[error] Missing required config key: data.tickers")
        sys.exit(1)
    if isinstance(tickers, str):
        print("[warn] data.tickers should be a list. Converting automatically.")
        tickers = [tickers]

    aux_tickers: List[str] = cfg_data.get("aux_tickers", [])
    start: str      = require(cfg_data, "start", "data")
    end: str        = require(cfg_data, "end", "data")
    buffer_size     = int(require(cfg_data, "buffer_size", "data"))
    data_dir        = Path(cfg_data.get("data_dir", "data"))

    # ── Strategy ──────────────────────────────────────────────────────────
    cfg_strategy    = require(cfg, "strategy", "root")
    strategy_type   = require(cfg_strategy, "type", "strategy")
    # _build_strategy() receives only the parameter kwargs
    strategy_params = {k: v for k, v in cfg_strategy.items() if k != "type"}

    # ── Risk ──────────────────────────────────────────────────────────────
    cfg_risk        = require(cfg, "risk", "root")
    initial_capital = float(require(cfg_risk, "initial_capital", "risk"))
    risk_fraction   = float(require(cfg_risk, "risk_fraction", "risk"))

    # ── Execution ─────────────────────────────────────────────────────────
    cfg_exec        = require(cfg, "execution", "root")
    commission_pct  = float(require(cfg_exec, "commission_pct", "execution"))
    slippage_bps    = float(require(cfg_exec, "slippage_bps", "execution"))

    # ── Analytics ─────────────────────────────────────────────────────────
    cfg_analytics   = require(cfg, "analytics", "root")
    trading_days    = int(require(cfg_analytics, "trading_days_per_year", "analytics"))
    cfg_filter = cfg.get("regime_filter")   # None for MA Crossover / TSMOM configs for now

    # ── Fetch data for every ticker (tradeable + aux) ─────────────────────
    for t in tickers + aux_tickers:
        fetch_data(
            ticker=t, start=start, end=end,
            data_dir=data_dir, refresh=args.refresh,
        )

    # ── Output directory & DB ─────────────────────────────────────────────
    out_dir = create_output_dir(run_name, tickers[0])
    db_path = str(out_dir / "backtest.db")

    if os.path.exists(db_path):
        os.remove(db_path)

    save_config_snapshot(cfg, out_dir)

    # ── Run backtest ──────────────────────────────────────────────────────
    # run_backtest() handles all component wiring internally and returns
    # a single-row Polars DataFrame of performance metrics.
    metrics = run_backtest(
        tickers=tickers,
        aux_tickers=aux_tickers,
        start=start,
        end=end,
        data_dir=str(data_dir),
        buffer_size=buffer_size,
        strategy_type=strategy_type,
        strategy_params=strategy_params,
        initial_capital=initial_capital,
        risk_fraction=risk_fraction,
        commission_pct=commission_pct,
        slippage_bps=slippage_bps,
        db_path=db_path,
        trading_days=trading_days,
        regime_filter_cfg=cfg_filter,
    )

    save_metrics_csv(metrics, out_dir)

    print(f"\n{'='*54}")
    print(f"  All artifacts → {out_dir}/")
    print(f"{'='*54}\n")


if __name__ == "__main__":
    main()
