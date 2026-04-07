# =============================================================
# src/data_execution.py
# CSVParquetDataHandler & SimulatedExecutionHandler
# =============================================================
from __future__ import annotations

import collections
import queue
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

from src.events import (
    DataHandlerBase,
    ExecutionHandlerBase,
    FillEvent,
    MarketEvent,
    OrderEvent,
)


# ──────────────────────────────────────────────────────────────────────────────
# CSVParquetDataHandler
# ──────────────────────────────────────────────────────────────────────────────

class CSVParquetDataHandler(DataHandlerBase):
    """
    Multi-ticker event-driven data handler.

    Loads all tickers (tradeable + aux) from CSV/Parquet, aligns them on
    their shared date intersection, and advances all rolling buffers in
    lockstep on each call to stream_next().

    ONLY tradeable `tickers` have MarketEvents with full OHLCV data.
    `aux_tickers` (e.g. VIX) are buffer-only [read via
    get_latest_bars() but unused in Portfolio or ExecutionHandler]

    Parameters
    ----------
    events      : Shared event queue.
    tickers     : Tradeable tickers; 1 MarketEvent per bar.
    data_dir    : Directory containing {ticker}.csv or {ticker}.parquet.
    start       : Inclusive start date "YYYY-MM-DD".
    end         : Inclusive end date "YYYY-MM-DD".
    buffer_size : Max rows per deque. Must exceed the largest indicator window.
    aux_tickers : Data-only tickers (e.g. ["VIX"])
    """

    def __init__(
        self,
        events: queue.Queue,
        tickers: List[str],
        data_dir: str,
        start: str,
        end: str,
        buffer_size: int,
        aux_tickers: Optional[List[str]] = None,
    ) -> None:
        self._events = events
        self._tickers: List[str] = list(tickers)
        self._aux_tickers: List[str] = list(aux_tickers or [])
        self._all_tickers: List[str] = self._tickers + self._aux_tickers
        self._buffer_size = buffer_size

        raw: Dict[str, pl.DataFrame] = {
            t: self._load_file(data_dir, t, start, end)
            for t in self._all_tickers
        }

        # Inner-join on date: only advance on days all tickers share.
        # Critical for aux tickers like VIX which have slightly different coverage.
        common_dates: set = set(raw[self._all_tickers[0]]["date"].to_list())
        for t in self._all_tickers[1:]:
            common_dates &= set(raw[t]["date"].to_list())

        self._aligned_dates: List[str] = sorted(common_dates)
        if not self._aligned_dates:
            raise ValueError(
                f"No overlapping dates found across tickers: {self._all_tickers}. "
                "Check date ranges and data files."
            )

        # Filter to aligned dates and sort once — enables O(1) positional access
        self._data: Dict[str, pl.DataFrame] = {
            t: raw[t]
            .filter(pl.col("date").is_in(self._aligned_dates))
            .sort("date")
            for t in self._all_tickers
        }

        self._buffers: Dict[str, collections.deque] = {
            t: collections.deque(maxlen=buffer_size) for t in self._all_tickers
        }
        self._current_idx: int = 0
        self._total_bars: int = len(self._aligned_dates)

    # ── Loading ───────────────────────────────────────────────────────────

    @staticmethod
    def _load_file(data_dir: str, ticker: str, start: str, end: str) -> pl.DataFrame:
        base = Path(data_dir) / ticker
        path_parquet = base.with_suffix(".parquet")
        path_csv = base.with_suffix(".csv")

        if path_parquet.exists():
            df = pl.read_parquet(path_parquet)
        elif path_csv.exists():
            df = pl.read_csv(path_csv, try_parse_dates=True)
        else:
            raise FileNotFoundError(
                f"No CSV or Parquet file found for ticker '{ticker}' in '{data_dir}'"
            )

        # Normalise date to string "YYYY-MM-DD" for consistent set intersection
        if df["date"].dtype != pl.Utf8:
            df = df.with_columns(pl.col("date").cast(pl.Utf8))

        return df.filter((pl.col("date") >= start) & (pl.col("date") <= end))

    # ── Core Advance Method ───────────────────────────────────────────────

    def stream_next(self) -> bool:
        """
        Advance one bar on the shared aligned timeline.

        Uses O(1) positional row access (not a date filter scan) to append
        each ticker's bar to its rolling deque. Outputs one MarketEvent
        per tradeable ticker, populated with full OHLCV data.

        Returns False when the data source is exhausted.
        """
        if self._current_idx >= self._total_bars:
            return False

        idx = self._current_idx

        # Update all buffers (tradeable and aux) before any events.
        # This guarantees that when a strategy reads an aux buffer (e.g. VIX)
        # in response to a MarketEvent, the data is already current.
        for t in self._all_tickers:
            row_df: pl.DataFrame = self._data[t][idx]
            self._buffers[t].append(row_df)

        # Output MarketEvent with full OHLCV for tradeable tickers only
        for t in self._tickers:
            bar: dict = self._data[t].row(idx, named=True)
            self._events.put(
                MarketEvent(
                    ticker=t,
                    timestamp=str(bar["date"]),
                    open=float(bar.get("open", bar.get("close", 0.0))),
                    high=float(bar.get("high", bar.get("close", 0.0))),
                    low=float(bar.get("low", bar.get("close", 0.0))),
                    close=float(bar["close"]),
                    volume=float(bar.get("volume", 0.0)),
                )
            )

        self._current_idx += 1
        return True

    def update_bars(self) -> bool:
        """Backward-compat alias for stream_next(). Prefer stream_next()."""
        return self.stream_next()

    # ── Buffer Access ─────────────────────────────────────────────────────

    def get_latest_bars(self, ticker: str, N: int = 1) -> Optional[pl.DataFrame]:
        """
        Return the last N bars for `ticker` as a Polars DataFrame.
        Works for both tradeable and aux tickers (e.g. VIX).
        Returns None during warm-up when fewer than N bars are buffered.
        Raises KeyError if `ticker` is not registered.
        """
        buf = self._buffers.get(ticker)
        if buf is None:
            raise KeyError(
                f"Ticker '{ticker}' is not registered. "
                f"Available: {self._all_tickers}"
            )
        if len(buf) < N:
            return None
        return pl.concat(list(buf)[-N:])

    def get_rolling_buffer(self, ticker: str) -> List[dict]:
        """
        Return the full rolling buffer as list of bar dicts.
        Prefer get_latest_bars() for indicator calculations [helps in debug]
        """
        buf = self._buffers.get(ticker)
        if buf is None:
            raise KeyError(
                f"Ticker '{ticker}' is not registered. "
                f"Available: {self._all_tickers}"
            )
        return [row_df.to_dicts()[0] for row_df in buf]

    def get_latest_bar(self, ticker: str) -> Optional[dict]:
        """Return the most recently consumed bar as a dict."""
        buf = self._buffers.get(ticker)
        if buf is None:
            raise KeyError(
                f"Ticker '{ticker}' is not registered. "
                f"Available: {self._all_tickers}"
            )
        if not buf:
            return None
        return buf[-1].to_dicts()[0]

    def get_current_datetime(self) -> Optional[str]:
        """Return the timestamp of the last output bar."""
        if self._current_idx == 0:
            return None
        return self._aligned_dates[self._current_idx - 1]


# ──────────────────────────────────────────────────────────────────────────────
# SimulatedExecutionHandler
# ──────────────────────────────────────────────────────────────────────────────

class SimulatedExecutionHandler(ExecutionHandlerBase):
    """
    Simulates order fills with flat commission % and basis-point slippage.

    Fill price:
      BUY  → market close + slippage  (adverse)
      SELL → market close - slippage  (adverse)

    FillEvent.cost:
      BUY  → positive  (cash outflow:  notional + commission)
      SELL → negative  (cash inflow: -(notional - commission))

    FillEvent.slippage: total currency-unit slippage (informational).

    Parameters
    ----------
    commission_pct : Flat rate applied to gross notional (e.g. 0.001 = 10 bps).
    slippage_bps   : One-way slippage in basis points (e.g. 5.0 = 5 bps).
    """

    def __init__(
        self,
        events: queue.Queue,
        data_handler: DataHandlerBase,
        commission_pct: float,
        slippage_bps: float,
    ) -> None:
        self._events = events
        self._data_handler = data_handler
        self._commission_pct = commission_pct
        self._slippage_bps = slippage_bps

    def execute_order(self, event: OrderEvent) -> None:
        bars = self._data_handler.get_latest_bars(event.ticker, N=1)
        if bars is None:
            return

        price: float = float(bars["close"][0])
        slippage_per_share: float = price * (self._slippage_bps / 10_000.0)

        if event.action == "BUY":
            fill_price: float = price + slippage_per_share
        else:
            fill_price: float = price - slippage_per_share

        notional: float = fill_price * abs(event.quantity)
        commission: float = notional * self._commission_pct
        slippage_cost: float = slippage_per_share * abs(event.quantity)

        if event.action == "BUY":
            cost: float = notional + commission      # cash outflow (positive)
        else:
            cost: float = -(notional - commission)   # cash inflow (negative)

        self._events.put(
            FillEvent(
                ticker=event.ticker,
                quantity=event.quantity,
                action=event.action,
                fill_price=fill_price,
                cost=cost,
                slippage=slippage_cost,
                timestamp=self._data_handler.get_current_datetime() or "",
            )
        )
