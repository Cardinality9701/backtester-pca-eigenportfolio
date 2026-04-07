# =============================================================
# src/engine_analytics.py
# Main Event Loop (_build_strategy, run_backtest) & Analytics Engine
# =============================================================
from __future__ import annotations

import queue
import datetime
import sqlite3
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from src.data_execution import CSVParquetDataHandler, SimulatedExecutionHandler
from src.events import (
    DataHandlerBase,
    EventQueue,
    ExecutionHandlerBase,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
    StrategyBase,
)
from src.portfolio_risk import Portfolio, RiskManager
from src.strategy import (
    BollingerBandStrategy,
    MovingAverageCrossStrategy,
    RSIStrategy,
    TSMOMStrategy,
    VIXRegimeFilter,
)
from eigenportfolio.strategy import EigenportfolioStrategy


# ---------------------------------------------------------------
# Strategy Factory
# ---------------------------------------------------------------

def _build_strategy(
    strategy_type: str,
    ticker: str,
    event_queue: queue.Queue,
    data_handler,
    regime_filter_cfg: Optional[Dict] = None,
    tickers: Optional[List[str]] = None,
    **strategy_params,
) -> StrategyBase:
    """
    Create a strategy by name, then optionally wrap it with
    VIXRegimeFilter if regime_filter_cfg is provided and enabled.
    """
    if strategy_type == "MovingAverageCross":
        strategy = MovingAverageCrossStrategy(
            ticker=ticker,
            event_queue=event_queue,
            data_handler=data_handler,
            fast_window=int(strategy_params["fast_window"]),
            slow_window=int(strategy_params["slow_window"]),
        )
    elif strategy_type == "TSMOM":
        strategy = TSMOMStrategy(
            ticker=ticker,
            event_queue=event_queue,
            data_handler=data_handler,
            lookback_window=int(strategy_params["lookback_window"]),
            skip_window=int(strategy_params["skip_window"]),
            vol_window=int(strategy_params["vol_window"]),
            vol_target=float(strategy_params["vol_target"]),
            rebalance_every=int(strategy_params["rebalance_every"]),
        )
    elif strategy_type == "BollingerBand":
        strategy = BollingerBandStrategy(
            ticker=ticker,
            event_queue=event_queue,
            data_handler=data_handler,
            window=int(strategy_params["window"]),
            num_std=float(strategy_params["num_std"]),
        )
    elif strategy_type == "RSI":
        strategy = RSIStrategy(
            ticker=ticker,
            event_queue=event_queue,
            data_handler=data_handler,
            rsi_period=int(strategy_params["rsi_period"]),
            oversold=float(strategy_params["oversold"]),
            overbought=float(strategy_params["overbought"]),
        )
    elif strategy_type == "Eigenportfolio":
        if tickers is None:
            raise ValueError(
                "EigenportfolioStrategy requires 'tickers' (full list). "
                "Pass tickers= to _build_strategy()."
            )
        # Portfolio-level strategy, skip VIX filter wrapper
        return EigenportfolioStrategy(
            tickers=tickers,
            event_queue=event_queue,
            data_handler=data_handler,
            **strategy_params,
        )
    else:
        raise ValueError(
            f"Unknown strategy_type: '{strategy_type}'. "
            f"Available: MovingAverageCross, TSMOM, BollingerBand, RSI, Eigenportfolio."
        )

    # Optionally wrap with VIX regime filter (per-instrument strategies only)
    if regime_filter_cfg and regime_filter_cfg.get("enabled", False):
        strategy = VIXRegimeFilter(
            inner=strategy,
            event_queue=event_queue,
            data_handler=data_handler,
            vix_ticker=str(regime_filter_cfg.get("vix_ticker", "VIX")),
            vix_threshold=float(regime_filter_cfg.get("vix_threshold", 20.0)),
            vix_smoothing_window=int(regime_filter_cfg.get("vix_smoothing_window", 5)),
            suppress_direction=str(regime_filter_cfg.get("suppress_direction", "LONG")),
            exit_on_suppress=bool(regime_filter_cfg.get("exit_on_suppress", True)),
        )

    return strategy


# ---------------------------------------------------------------
# Main Event Loop
# ---------------------------------------------------------------

def run_backtest(
    tickers: List[str],
    aux_tickers: List[str],
    start: str,
    end: str,
    data_dir: str,
    buffer_size: int,
    strategy_type: str,
    strategy_params: dict,
    initial_capital: float,
    risk_fraction: float,
    commission_pct: float,
    slippage_bps: float,
    db_path: str,
    trading_days: int = 252,
    regime_filter_cfg: Optional[dict] = None,
) -> pl.DataFrame:
    """
    Wire up all components and run the full event loop.

    Event routing:
      MarketEvent  → strategies (by ticker or portfolio-wide) + portfolio.update_market()
      SignalEvent  → risk_manager.size_order()  (outputs OrderEvent)
      OrderEvent   → execution_handler.execute_order()  (outputs FillEvent)
      FillEvent    → portfolio.update_fill()

    Returns a single-row Polars DataFrame of performance metrics.
    """
    events: EventQueue = queue.Queue()

    data_handler = CSVParquetDataHandler(
        events=events,
        tickers=tickers,
        data_dir=data_dir,
        start=start,
        end=end,
        buffer_size=buffer_size,
        aux_tickers=aux_tickers,
    )

    # Portfolio-level strategy: one instance handles all tickers simultaneously.
    # Per-instrument strategies: one instance per ticker, each fires independently.
    if strategy_type == "Eigenportfolio":
        strategies = [
            _build_strategy(
                strategy_type     = strategy_type,
                ticker            = tickers[0],
                tickers           = tickers,
                event_queue       = events,
                data_handler      = data_handler,
                regime_filter_cfg = regime_filter_cfg,
                **strategy_params,
            )
        ]
    else:
        strategies = [
            _build_strategy(
                strategy_type     = strategy_type,
                ticker            = t,
                tickers           = None,
                event_queue       = events,
                data_handler      = data_handler,
                regime_filter_cfg = regime_filter_cfg,
                **strategy_params,
            )
            for t in tickers
        ]

    if not strategies:
        raise RuntimeError(f"No strategies built for type '{strategy_type}'.")

    portfolio = Portfolio(
        initial_capital=initial_capital,
        db_path=db_path,
    )

    risk_manager = RiskManager(
        event_queue=events,
        portfolio=portfolio,
        data_handler=data_handler,
        risk_fraction=risk_fraction,
    )

    execution_handler = SimulatedExecutionHandler(
        events=events,
        data_handler=data_handler,
        commission_pct=commission_pct,
        slippage_bps=slippage_bps,
    )

    # ── Main Event Loop ────────────────────────────────────────────────────
    while True:
        if not data_handler.stream_next():
            break

        while not events.empty():
            event = events.get(block=False)

            if isinstance(event, MarketEvent):
                for strat in strategies:
                    if hasattr(strat, "on_market"):
                        # Portfolio-level strategy: receives every ticker's
                        # market event to build its internal price matrix
                        strat.on_market(event)
                    elif strat.ticker == event.ticker:
                        # Per-instrument strategy: only fires on its own ticker
                        strat.calculate_signals(event)
                portfolio.update_market(event)

            elif isinstance(event, SignalEvent):
                risk_manager.size_order(event)

            elif isinstance(event, OrderEvent):
                execution_handler.execute_order(event)

            elif isinstance(event, FillEvent):
                portfolio.update_fill(event)

    portfolio.flush_all()

    return compute_metrics(db_path=db_path, trading_days=trading_days)


# ---------------------------------------------------------------
# Analytics Engine
# ---------------------------------------------------------------

def compute_metrics(
    db_path: str,
    trading_days: int,
) -> pl.DataFrame:
    """
    Read the SQLite output and compute performance metrics.

    Metrics:
      1. Sharpe Ratio      — annualised, 0% RFR
      2. Max Drawdown (%)  — peak-to-trough of the equity curve
      3. Calmar Ratio      — annualised geometric return / |max drawdown|
      4. Turnover          — one-sided: total SELL notional / mean equity
      5. Hit Rate (%)      — % of closing (SELL) trades with pnl > 0
      6. Ann. Return (%)   — geometric annualised return
      7. Final Equity      — terminal portfolio value

    trading_days MUST come from config.analytics.trading_days_per_year.
    """
    conn = sqlite3.connect(db_path)

    def _query(sql: str) -> pl.DataFrame:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)})

    equity_df = _query(
        "SELECT timestamp, equity FROM equity_curve ORDER BY id"
    )


    trades_df = _query(
        "SELECT action, fill_price, quantity, realised_pnl FROM trades"
    )
    conn.close()

    if equity_df.is_empty():
        raise RuntimeError("Equity curve is empty? Did backtest complete?")

    equity_df = (
      equity_df
      .with_columns(pl.col("timestamp").str.slice(0, 10).alias("date"))
      .group_by("date")
      .agg(pl.col("equity").last())   # keep final mark-to-market of each trading day
      .sort("date")
    )

    equities: np.ndarray = equity_df["equity"].to_numpy(allow_copy=True)
    returns: np.ndarray = np.diff(equities) / equities[:-1]

    # 1. Sharpe Ratio
    ret_std = float(returns.std())
    sharpe: float = (
        float(returns.mean() / ret_std) * np.sqrt(trading_days)
        if ret_std > 1e-10 else 0.0
    )

    # 2. Maximum Drawdown
    peak: np.ndarray = np.maximum.accumulate(equities)
    drawdowns: np.ndarray = (equities - peak) / peak
    max_drawdown: float = float(drawdowns.min())

    # 3. Annualised Geometric Return & Calmar
    dates   = equity_df["date"].to_list()
    d0      = datetime.date.fromisoformat(str(dates[0])[:10])
    d1      = datetime.date.fromisoformat(str(dates[-1])[:10])
    n_years = (d1 - d0).days / 365.25
    ann_return: float = (
        float((equities[-1] / equities[0]) ** (1.0 / n_years) - 1.0)
        if n_years > 0 and equities[0] > 0.0 else 0.0
    )
    calmar: float = (
        ann_return / abs(max_drawdown)
        if abs(max_drawdown) > 1e-10 else float("inf")
    )

    # 4. One-sided Turnover & 5. Hit Rate
    turnover: float = 0.0
    hit_rate: float = 0.0
    n_closing: int = 0

    if not trades_df.is_empty():
        sell_mask = trades_df["action"] == "SELL"
        sell_notional: float = float(
            (trades_df.filter(sell_mask)["fill_price"]
             * trades_df.filter(sell_mask)["quantity"]).sum()
        )
        avg_equity: float = float(np.mean(equities))
        turnover = sell_notional / avg_equity if avg_equity > 0.0 else 0.0

        closing = trades_df.filter(pl.col("realised_pnl") != 0.0)
        n_closing = len(closing)
        if n_closing > 0:
            n_winners = int((closing["realised_pnl"] > 0.0).sum())
            hit_rate = n_winners / n_closing

    metrics = pl.DataFrame({
        "sharpe_ratio":     [round(sharpe, 4)],
        "max_drawdown_pct": [round(max_drawdown * 100, 4)],
        "calmar_ratio":     [round(calmar, 4)],
        "turnover":         [round(turnover, 4)],
        "hit_rate_pct":     [round(hit_rate * 100, 2)],
        "total_trades":     [n_closing],
        "ann_return_pct":   [round(ann_return * 100, 4)],
        "final_equity":     [round(float(equities[-1]), 2)],
    })

    print("\n── Analytics Results ──────────────────────────")
    print(metrics)
    return metrics
