# =============================================================
# src/portfolio_risk.py
# Portfolio, RiskManager, & SQLite persistence layer
# =============================================================
from __future__ import annotations

import sqlite3
from typing import Dict, List, Literal, Optional, Tuple

from src.events import (
    DataHandlerBase,
    EventQueue,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)


# ---------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT    NOT NULL,
    ticker       TEXT    NOT NULL,
    action       TEXT    NOT NULL,
    quantity     INTEGER NOT NULL,
    fill_price   REAL    NOT NULL,
    cost         REAL    NOT NULL,
    slippage     REAL    NOT NULL,
    realised_pnl REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS equity_curve (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    equity    REAL    NOT NULL
);
"""

_INSERT_TRADE = """
    INSERT INTO trades
        (timestamp, ticker, action, quantity, fill_price, cost, slippage, realised_pnl)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_EQUITY = """
    INSERT INTO equity_curve (timestamp, equity) VALUES (?, ?)
"""


# ---------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------

class Portfolio:
    """
    Tracks cash, open positions, and average entry prices.
    Streams trade records and equity snapshots to SQLite via buffered
    executemany calls — never holds the full history in RAM.

    Cash accounting convention (aligns with FillEvent.cost sign convention):
      BUY  → cost is positive  → self._cash -= cost  (outflow)
      SELL → cost is negative  → self._cash -= cost  (inflow, since cost < 0)

    This means a single line handles both cases:
        self._cash -= event.cost

    db_path is required [passed explicitly from the resolved
    output directory; no default to prevent stray DB files]
    """

    def __init__(
        self,
        initial_capital: float,
        db_path: str,
        flush_every: int = 50,
    ) -> None:
        self._cash: float = initial_capital
        self._initial_capital: float = initial_capital

        self._positions: Dict[str, int] = {}
        self._avg_entry: Dict[str, float] = {}

        # Price cache: updated on every MarketEvent so equity snapshots
        # reflect ALL open positions, not just the ticker that just fired.
        self._last_price: Dict[str, float] = {}
        # Timestamp guard: ensures one equity snapshot per bar, not one
        # per tradeable ticker (which would double/triple-count on multi-ticker).
        self._last_snapshot_ts: str = ""

        self._conn: sqlite3.Connection = sqlite3.connect(
            db_path, check_same_thread=False
        )
        for stmt in _DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)
        self._conn.commit()

        self._flush_every = flush_every
        self._trade_buf: List[Tuple] = []
        self._equity_buf: List[Tuple] = []

    # ── Market Update ─────────────────────────────────────────────────────

    def update_market(self, event: MarketEvent) -> None:
        """
        Called for every MarketEvent (one per tradeable ticker per bar).

        Always updates the price cache so equity is current for all tickers.
        Writes one equity snapshot per unique timestamp; guard
        prevents duplicate rows when multiple tickers fire on the same bar.
        """
        # Update price cache regardless of whether we snapshot this tick
        self._last_price[event.ticker] = event.close

        # Skip snapshot if this bar's timestamp has already been recorded
        if event.timestamp == self._last_snapshot_ts:
            return
        self._last_snapshot_ts = event.timestamp

        # Mark-to-market equity across ALL open positions
        equity: float = self._cash + sum(
            qty * self._last_price.get(t, 0.0)
            for t, qty in self._positions.items()
        )
        self._equity_buf.append((event.timestamp, equity))
        if len(self._equity_buf) >= self._flush_every:
            self._flush_equity()

    # ── Fill Update ───────────────────────────────────────────────────────

    def update_fill(self, event: FillEvent) -> None:
        ticker = event.ticker
        qty_delta: int = event.quantity if event.action == "BUY" else -event.quantity
        prev_pos: int = self._positions.get(ticker, 0)
        new_pos: int = prev_pos + qty_delta

        realised_pnl: float = self._compute_realised_pnl(ticker, event, prev_pos)

        # Single formula handles both directions:
        #   BUY : cost > 0 → cash decreases (outflow)
        #   SELL: cost < 0 → cash increases (inflow)
        self._cash -= event.cost

        if new_pos == 0:
            self._positions.pop(ticker, None)
            self._avg_entry.pop(ticker, None)
        else:
            self._positions[ticker] = new_pos
            # Recompute avg entry only when the position is growing (opening leg)
            if abs(new_pos) > abs(prev_pos):
                prev_avg: float = self._avg_entry.get(ticker, 0.0)
                self._avg_entry[ticker] = (
                    abs(prev_pos) * prev_avg + abs(qty_delta) * event.fill_price
                ) / abs(new_pos)

        self._trade_buf.append((
            event.timestamp, ticker, event.action, event.quantity,
            event.fill_price, event.cost, event.slippage, realised_pnl,
        ))
        if len(self._trade_buf) >= self._flush_every:
            self._flush_trades()

    def _compute_realised_pnl(
        self, ticker: str, event: FillEvent, prev_pos: int
    ) -> float:
        """
        Realised P&L on the closing leg only (opening legs return 0.0).

        Commission extraction differs by direction due to the FillEvent.cost
        sign convention established in SimulatedExecutionHandler:
          BUY : cost =  notional + commission → commission = cost - notional
          SELL: cost = -(notional - commission) → commission = cost + notional
        """
        notional: float = event.fill_price * event.quantity
        if event.action == "BUY":
            commission: float = event.cost - notional
        else:
            commission: float = event.cost + notional

        avg_entry: float = self._avg_entry.get(ticker, event.fill_price)

        if event.action == "SELL" and prev_pos > 0:
            closed_qty: int = min(prev_pos, event.quantity)
            return (event.fill_price - avg_entry) * closed_qty - commission

        if event.action == "BUY" and prev_pos < 0:
            closed_qty: int = min(abs(prev_pos), event.quantity)
            return (avg_entry - event.fill_price) * closed_qty - commission

        return 0.0

    # ── SQLite I/O ────────────────────────────────────────────────────────

    def _flush_trades(self) -> None:
        if self._trade_buf:
            self._conn.executemany(_INSERT_TRADE, self._trade_buf)
            self._conn.commit()
            self._trade_buf.clear()

    def _flush_equity(self) -> None:
        if self._equity_buf:
            self._conn.executemany(_INSERT_EQUITY, self._equity_buf)
            self._conn.commit()
            self._equity_buf.clear()

    def flush_all(self) -> None:
        """Force-flush all remaining buffers and close DB connection."""
        self._flush_trades()
        self._flush_equity()
        self._conn.close()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> Dict[str, int]:
        return dict(self._positions)


# ---------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------

class RiskManager:
    """
    Sizes a SignalEvent into one or more OrderEvents using fixed-fractional
    sizing scaled by signal strength:

        target_notional = current_cash * risk_fraction * signal.strength
        quantity        = floor(target_notional / close_price)

    signal.strength ∈ (0, 1] — strategies like RSI pass proportional values
    so stronger signals get proportionally larger positions.

    Enforces a flat-position rule: only one open position per ticker.
    On direction reversal, the existing position is closed first via a
    separate OrderEvent before the new position is opened.

    risk_fraction required; must come from config.risk.risk_fraction.
    No default to prevent silent misconfiguration of position sizing.
    """

    def __init__(
        self,
        event_queue: EventQueue,
        portfolio: Portfolio,
        data_handler: DataHandlerBase,
        risk_fraction: float,
    ) -> None:
        self._queue = event_queue
        self._portfolio = portfolio
        self._data_handler = data_handler
        self._risk_fraction = risk_fraction

    def size_order(self, event: SignalEvent) -> None:
        bar = self._data_handler.get_latest_bar(event.ticker)
        if bar is None:
            return

        price: float = float(bar["close"])
        if price <= 0.0:
            return

        # Signal strength scales the notional — EXIT signals carry strength=0.0
        # so they never accidentally open a position through this path
        strength: float = max(0.0, min(event.strength, 1.0))
        target_notional: float = self._portfolio.cash * self._risk_fraction * strength
        quantity: int = max(1, int(target_notional / price))
        current_pos: int = self._portfolio.positions.get(event.ticker, 0)

        if event.direction == "LONG" and current_pos <= 0:
            if current_pos < 0:
                # Close existing short before opening long
                self._emit_order(event.ticker, abs(current_pos), "BUY")
            self._emit_order(event.ticker, quantity, "BUY")

        elif event.direction == "SHORT" and current_pos >= 0:
            if current_pos > 0:
                # Close existing long before opening short
                self._emit_order(event.ticker, current_pos, "SELL")
            self._emit_order(event.ticker, quantity, "SELL")

        elif event.direction == "EXIT" and current_pos != 0:
            action: Literal["BUY", "SELL"] = "SELL" if current_pos > 0 else "BUY"
            self._emit_order(event.ticker, abs(current_pos), action)

    def _emit_order(
        self, ticker: str, quantity: int, action: Literal["BUY", "SELL"]
    ) -> None:
        self._queue.put(OrderEvent(ticker=ticker, quantity=quantity, action=action))
