# =============================================================
# src/events.py
# Events & Abstract Base Classes
# =============================================================
from __future__ import annotations

import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import polars as pl

# Alias used throughout the engine
EventQueue = queue.Queue

# ---------------------------------------------------------------
# Event Dataclasses
# ---------------------------------------------------------------

@dataclass
class MarketEvent:
    """
    Used for DataHandler on each new bar for each tradeable ticker
    NOTE: Aux tickers (e.g. VIX) do NOT output MarketEvents [updated silently 
    and readable via get_latest_bars()]
    """
    ticker: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    type: str = field(default="MARKET", init=False, repr=False)


@dataclass
class SignalEvent:
    """
    Used by Strategy when a signal condition is met.
    direction : 'LONG' | 'SHORT' | 'EXIT'
    strength  : Scalar in (0, 1]. Used by RiskManager for position sizing.
                EXIT signals should always carry strength=0.0.
    """
    ticker: str
    timestamp: str
    direction: Literal["LONG", "SHORT", "EXIT"]
    strength: float = 1.0
    type: str = field(default="SIGNAL", init=False, repr=False)


@dataclass
class OrderEvent:
    """
    Used by RiskManager after sizing a SignalEvent into shares.
    action: 'BUY' | 'SELL'
    """
    ticker: str
    quantity: int
    action: Literal["BUY", "SELL"]
    type: str = field(default="ORDER", init=False, repr=False)


@dataclass
class FillEvent:
    """
    Used by ExecutionHandler after simulating a market fill.
    cost     : Total cash outflow/inflow (notional ± commission).
    slippage : Currency-unit slippage incurred (informational only).
    """
    ticker: str
    quantity: int
    action: Literal["BUY", "SELL"]
    fill_price: float
    cost: float
    slippage: float
    timestamp: str
    type: str = field(default="FILL", init=False, repr=False)


# ---------------------------------------------------------------
# Abstract Base Classes
# ---------------------------------------------------------------

class DataHandlerBase(ABC):

    @abstractmethod
    def stream_next(self) -> bool:
        """
        Advance the internal cursor by one bar.
        - Updates rolling buffers for ALL tickers (tradeable + aux).
        - Outputs one MarketEvent per tradeable ticker only.
        Returns False when the data source is exhausted.
        """
        ...

    @abstractmethod
    def get_latest_bars(self, ticker: str, N: int = 1) -> Optional[pl.DataFrame]:
        """
        Return the last N bars for `ticker` as a Polars DataFrame.
        Works for both tradeable and aux tickers (e.g. VIX).
        Returns None during warm-up when fewer than N bars are buffered.
        Raises KeyError if `ticker` is not registered.
        """
        ...

    @abstractmethod
    def get_rolling_buffer(self, ticker: str) -> List[Dict]:
        """
        Return the full current rolling buffer as a list of bar dicts.
        Prefer get_latest_bars() for indicator calculations — this method
        is primarily useful for debugging and inspection.
        """
        ...

    @abstractmethod
    def get_latest_bar(self, ticker: str) -> Optional[Dict]:
        """Return the most recently consumed bar as a dict."""
        ...


class StrategyBase(ABC):
    """
    Interface contract for all trading strategies.

    Shared workflow pattern: ticker, event queue, and data handler 
    set in __init__; subclasses only need to implement calculate_signals().
    """

    def __init__(
        self,
        ticker: str,
        events: EventQueue,
        data_handler: DataHandlerBase,
    ) -> None:
        self.ticker = ticker
        self._events = events
        self._data_handler = data_handler

    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> None:
        """
        Inspect a MarketEvent and optionally output SignalEvents.
        Must guard on event.ticker == self.ticker to avoid cross-ticker noise.
        """
        ...

class ExecutionHandlerBase(ABC):
    """Interface contract for all execution handlers."""

    @abstractmethod
    def execute_order(self, event: OrderEvent) -> None:
        """Simulate a fill for an OrderEvent and output a FillEvent."""
        ...
