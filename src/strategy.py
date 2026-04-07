# =============================================================
# src/strategy.py
# Strategy implementations:
#   1. MovingAverageCrossStrategy  — dual SMA crossover
#   2. TSMOMStrategy               — Time-Series Momentum (MOP 2012)
#   3. BollingerBandStrategy       — mean-reversion Bollinger Bands
#   4. RSIStrategy                 — mean-reversion RSI (Wilder)
# =============================================================
from __future__ import annotations

from typing import Optional
from abc import ABC, abstractmethod
import polars as pl

import numpy as np
import queue as _stdlib_queue  # aliased to avoid collision with module-level `queue`

from src.events import (
    DataHandlerBase,
    EventQueue,
    MarketEvent,
    SignalEvent,
    StrategyBase,
)


# ---------------------------------------------------------------
# Strategy 1: Moving Average Crossover
# ---------------------------------------------------------------

class MovingAverageCrossStrategy(StrategyBase):
    """
    Dual SMA crossover strategy.

    Signal logic:
      fast crosses ABOVE slow → LONG
      fast crosses BELOW slow → SHORT

    Signal strength = raw MA spread (in price units). RiskManager clamps
    this to [0, 1], so MA Cross effectively uses flat sizing. A normalised
    strength could be added later if desired.

    Parameters
    ----------
    fast_window : Bars for the fast SMA. Must be < slow_window.
    slow_window : Bars for the slow SMA.
    """

    def __init__(
        self,
        ticker: str,
        event_queue: EventQueue,
        data_handler: DataHandlerBase,
        fast_window: int = 10,
        slow_window: int = 40,
    ) -> None:
        if fast_window >= slow_window:
            raise ValueError(
                f"fast_window ({fast_window}) must be < slow_window ({slow_window})."
            )
        super().__init__(ticker=ticker, events=event_queue, data_handler=data_handler)
        self._fast_window = fast_window
        self._slow_window = slow_window
        self._prev_fast_above: Optional[bool] = None

    def calculate_signals(self, event: MarketEvent) -> None:
        if event.ticker != self.ticker:
            return

        # get_latest_bars returns None during warm-up — handles the buffer guard
        buf = self._data_handler.get_latest_bars(self.ticker, N=self._slow_window + 1)
        if buf is None:
            return

        closes: np.ndarray = buf["close"].cast(float).to_numpy()
        fast_ma: float = float(np.mean(closes[-self._fast_window:]))
        slow_ma: float = float(np.mean(closes[-self._slow_window:]))
        curr_fast_above: bool = fast_ma > slow_ma

        # Warm-up: seed state on the first valid bar without giving a signal
        if self._prev_fast_above is None:
            self._prev_fast_above = curr_fast_above
            return

        if curr_fast_above and not self._prev_fast_above:
            self._events.put(SignalEvent(
                ticker=self.ticker,
                timestamp=event.timestamp,
                direction="LONG",
                strength=round(fast_ma - slow_ma, 6),
            ))
        elif not curr_fast_above and self._prev_fast_above:
            self._events.put(SignalEvent(
                ticker=self.ticker,
                timestamp=event.timestamp,
                direction="SHORT",
                strength=round(slow_ma - fast_ma, 6),
            ))

        self._prev_fast_above = curr_fast_above


# ---------------------------------------------------------------
# Strategy 2: Time-Series Momentum (TSMOM)
# ---------------------------------------------------------------

class TSMOMStrategy(StrategyBase):
    """
    Time-Series Momentum strategy.
    Reference: Moskowitz, Ooi & Pedersen (2012) — "Time Series Momentum"
               Journal of Financial Economics, 104(2), 228–250.

    Formation return:
        close[t - skip_window] / close[t - lookback_window] - 1

    Signal:
        formation_return > 0  →  LONG
        formation_return < 0  →  SHORT

    Signal strength is vol-scaled: vol_target / realised_vol (EWMA).
    RiskManager multiplies target_notional by signal.strength, so a
    high-vol period automatically reduces position size.

    Parameters
    ----------
    lookback_window : Total lookback in bars (default 252 ≈ 12 months).
    skip_window     : Near-end bars to skip to avoid reversal (default 21).
                      Must be < lookback_window.
    vol_window      : Bars for EWMA vol estimation (default 60).
    vol_target      : Annualised target volatility (default 0.15 = 15%).
    rebalance_every : Bars between forced rebalance signals (default 21).
    """

    def __init__(
        self,
        ticker: str,
        event_queue: EventQueue,
        data_handler: DataHandlerBase,
        lookback_window: int = 252,
        skip_window: int = 21,
        vol_window: int = 60,
        vol_target: float = 0.15,
        rebalance_every: int = 21,
    ) -> None:
        if skip_window >= lookback_window:
            raise ValueError(
                f"skip_window ({skip_window}) must be < lookback_window ({lookback_window})."
            )
        super().__init__(ticker=ticker, events=event_queue, data_handler=data_handler)
        self._lookback = lookback_window
        self._skip = skip_window
        self._vol_window = vol_window
        self._vol_target = vol_target
        self._rebalance_every = rebalance_every
        self._min_bars: int = lookback_window + 1
        self._prev_direction: Optional[str] = None
        self._bars_since_rebalance: int = 0

    def calculate_signals(self, event: MarketEvent) -> None:
        if event.ticker != self.ticker:
            return

        buf = self._data_handler.get_latest_bars(self.ticker, N=self._min_bars)
        if buf is None:
            return

        closes: np.ndarray = buf["close"].cast(float).to_numpy()

        # Formation return: skip most recent `skip_window` bars
        price_now: float = closes[-(self._skip + 1)]
        price_lookback: float = closes[-(self._lookback + 1)]
        if price_lookback <= 0.0:
            return
        formation_return: float = price_now / price_lookback - 1.0

        # EWMA realised volatility
        vol_closes = closes[-(self._vol_window + 1):]
        daily_returns: np.ndarray = np.diff(vol_closes) / vol_closes[:-1]
        weights: np.ndarray = np.exp(np.linspace(-1.0, 0.0, len(daily_returns)))
        weights /= weights.sum()
        ewma_mean: float = float(np.dot(weights, daily_returns))
        ewma_var: float = float(np.dot(weights, (daily_returns - ewma_mean) ** 2))
        realised_vol_ann: float = float(np.sqrt(ewma_var)) * np.sqrt(252)

        if realised_vol_ann < 1e-6:
            return

        vol_scale: float = round(float(self._vol_target / realised_vol_ann), 6)
        direction: str = "LONG" if formation_return > 0 else "SHORT"

        direction_changed: bool = direction != self._prev_direction
        rebalance_due: bool = self._bars_since_rebalance >= self._rebalance_every

        if direction_changed or rebalance_due:
            self._events.put(SignalEvent(
                ticker=self.ticker,
                timestamp=event.timestamp,
                direction=direction,
                strength=vol_scale,
            ))
            self._prev_direction = direction
            self._bars_since_rebalance = 0
        else:
            self._bars_since_rebalance += 1


# ---------------------------------------------------------------
# Strategy 3: Bollinger Band (mean-reversion)
# ---------------------------------------------------------------

class BollingerBandStrategy(StrategyBase):
    """
    Mean-reversion Bollinger Band strategy.

    Signal logic:
      close < lower band  →  LONG  (price below mean — buy the dip)
      close > upper band  →  EXIT  (price above mean — close long)

    `_position` state prevents redundant signal emission on consecutive
    bars where the condition holds. This keeps the trade log clean and
    avoids over-trading.

    Parameters
    ----------
    window  : Rolling lookback for mean/std (bars). Default 20.
    num_std : Band width in standard deviations. Default 2.0.
    """

    def __init__(
        self,
        ticker: str,
        event_queue: EventQueue,
        data_handler: DataHandlerBase,
        window: int = 20,
        num_std: float = 2.0,
    ) -> None:
        super().__init__(ticker=ticker, events=event_queue, data_handler=data_handler)
        self._window = window
        self._num_std = num_std
        self._position: str = "FLAT"

    def calculate_signals(self, event: MarketEvent) -> None:
        if event.ticker != self.ticker:
            return

        buf = self._data_handler.get_latest_bars(self.ticker, N=self._window + 1)
        if buf is None:
            return

        closes: np.ndarray = buf["close"].cast(float).to_numpy()
        window_closes = closes[-self._window:]
        mean: float = float(np.mean(window_closes))
        std: float = float(np.std(window_closes, ddof=1))

        if std < 1e-10:
            return

        upper: float = mean + self._num_std * std
        lower: float = mean - self._num_std * std
        current_price: float = closes[-1]

        if current_price < lower and self._position != "LONG":
            self._position = "LONG"
            self._events.put(SignalEvent(
                ticker=self.ticker,
                timestamp=event.timestamp,
                direction="LONG",
                strength=1.0,
            ))
        elif current_price > upper and self._position != "FLAT":
            self._position = "FLAT"
            self._events.put(SignalEvent(
                ticker=self.ticker,
                timestamp=event.timestamp,
                direction="EXIT",
                strength=0.0,
            ))


# ---------------------------------------------------------------
# Strategy 4: RSI (mean-reversion)
# ---------------------------------------------------------------

class RSIStrategy(StrategyBase):
    """
    Mean-reversion RSI strategy using Wilder's smoothed RSI.

    Signal logic:
      RSI < oversold   →  LONG  (momentum exhausted to downside)
      RSI > overbought →  EXIT  (momentum exhausted to upside)

    Signal strength ∈ (0, 1] is proportional to how far RSI is below
    the oversold threshold. Stronger oversold → larger position via
    RiskManager's strength-scaled notional.

    Parameters
    ----------
    rsi_period  : Lookback window for RSI calculation (bars). Default 14.
    oversold    : RSI threshold to go LONG. Default 30.0.
    overbought  : RSI threshold to EXIT. Default 70.0.
    """

    def __init__(
        self,
        ticker: str,
        event_queue: EventQueue,
        data_handler: DataHandlerBase,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        super().__init__(ticker=ticker, events=event_queue, data_handler=data_handler)
        self._rsi_period = rsi_period
        self._oversold = oversold
        self._overbought = overbought
        self._position: str = "FLAT"

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int) -> float:
        """
        Wilder's smoothed RSI. Requires len(closes) >= period + 1.

        Steps:
          1. Seed avg_gain / avg_loss with a simple mean over the first
             `period` deltas.
          2. Apply Wilder's EMA (alpha = 1/period) for subsequent bars.

        Returns a single float in [0, 100].
        """
        deltas: np.ndarray = np.diff(closes)
        gains: np.ndarray = np.where(deltas > 0.0, deltas, 0.0)
        losses: np.ndarray = np.where(deltas < 0.0, -deltas, 0.0)

        avg_gain: float = float(np.mean(gains[:period]))
        avg_loss: float = float(np.mean(losses[:period]))

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0.0:
            return 100.0
        rs: float = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def calculate_signals(self, event: MarketEvent) -> None:
        if event.ticker != self.ticker:
            return

        buf = self._data_handler.get_latest_bars(self.ticker, N=self._rsi_period + 1)
        if buf is None:
            return

        closes: np.ndarray = buf["close"].cast(float).to_numpy()
        rsi: float = self._compute_rsi(closes, self._rsi_period)

        if rsi < self._oversold and self._position != "LONG":
            self._position = "LONG"
            # More oversold → strength closer to 1.0
            strength: float = 1.0 - (rsi / self._oversold)
            self._events.put(SignalEvent(
                ticker=self.ticker,
                timestamp=event.timestamp,
                direction="LONG",
                strength=round(strength, 6),
            ))
        elif rsi > self._overbought and self._position != "FLAT":
            self._position = "FLAT"
            self._events.put(SignalEvent(
                ticker=self.ticker,
                timestamp=event.timestamp,
                direction="EXIT",
                strength=0.0,
            ))

# ---------------------------------------------------------------
#  Eigenportfoliio
# ---------------------------------------------------------------
class PortfolioStrategy(ABC):
    """
    Base class for strategies that produce portfolio-level weight vectors
    rather than per-instrument scalar signals.

    Subclasses implement compute_weights() which receives the full
    price matrix for all tickers and returns a Dict[str, float] of
    target weights. The engine calls this only when ALL tickers have
    a fresh bar — not per-instrument.
    """

    def __init__(
        self,
        tickers: List[str],
        event_queue: queue.Queue,
        data_handler: "CSVParquetDataHandler",
        rebalance_every: int = 21,
    ) -> None:
        self.tickers        = tickers
        self.event_queue    = event_queue
        self.data_handler   = data_handler
        self.rebalance_every = rebalance_every
        self._bar_count     = 0
        self._last_weights: Dict[str, float] = {t: 0.0 for t in tickers}

    @property
    def ticker(self) -> str:
      return self.tickers[0]

    @abstractmethod
    def compute_weights(
        self,
        price_matrix: np.ndarray,   # shape (lookback, n_tickers)
        tickers: List[str],
    ) -> Dict[str, float]:
        """
        Return target portfolio weights for each ticker.
        Weights must sum to ≤ 1.0 in absolute value.
        """
        raise NotImplementedError

    def on_market(self, event: "MarketEvent") -> None:
        """
        Called once per bar (on primary ticker's MarketEvent only).
        Increments bar count and fires rebalance on schedule.
        """
        if event.ticker != self.tickers[0]:
            return   # Only fire on primary ticker — avoids n_ticker calls per bar

        self._bar_count += 1
        if self._bar_count % self.rebalance_every != 0:
            return

        # Build price matrix from data handler buffer
        price_matrix, valid = self._build_price_matrix()
        if not valid:
            return   # Not enough history yet

        weights = self.compute_weights(price_matrix, self.tickers)
        self._last_weights = weights

        # One SignalEvent per ticker with weight as strength
        for ticker, weight in weights.items():
            if abs(weight) < 1e-6:
                direction = "EXIT"
                strength  = 0.0
            else:
                direction = "LONG" if weight > 0 else "SHORT"
                strength  = abs(weight)

            self.event_queue.put(SignalEvent(
                ticker    = ticker,
                timestamp = event.timestamp,
                direction = direction,
                strength  = strength,
            ))

    def _build_price_matrix(self) -> Tuple[np.ndarray, bool]:
        """
        Pull close prices for all tickers from the data handler buffer.
        Returns (matrix, is_valid). Matrix shape: (n_bars, n_tickers).
        """
        cols = []
        for t in self.tickers:
            buf = self.data_handler.get_rolling_buffer(t)   # returns List[Dict] or np.ndarray
            if buf is None or len(buf) < 2:
                return np.array([]), False
            closes = np.array([row["close"] for row in buf], dtype=float)
            cols.append(closes)

        min_len = min(len(c) for c in cols)
        matrix  = np.column_stack([c[-min_len:] for c in cols])
        return matrix, True

# ---------------------------------------------------------------
#  VIX Regime Filter Decorator
# ---------------------------------------------------------------
class VIXRegimeFilter(StrategyBase):
    """
    Regime-filter decorator that wraps any StrategyBase strategy.

    The inner strategy always runs so its internal state (MAs, RSI level,
    TSMOM formation return) stays current. Its output is intercepted via a
    staging queue; signals are either forwarded/suppressed/replaced with EXIT,
    depends on current VIX regime.

    NOTE: Skipping inner.calculate_signals() during risk-off would cause stale
    state on regime re-entry (e.g. MA Cross would re-seed _prev_fast_above
    incorrectly).

    Parameters
    ----------
    inner                : Any StrategyBase instance to wrap.
    vix_ticker           : Ticker name for VIX in DataHandler (default "VIX").
    vix_threshold        : VIX level above which regime is risk-off (default 20.0).
                           Chosen a priori at the long-run VIX average — not
                           optimised — to avoid overfitting a risk management knob.
    vix_smoothing_window : Bars to average VIX over to reduce single-bar spike
                           noise (default 5). A spike to VIX=22 for one bar
                           should not trigger a full position exit.
    suppress_direction   : "LONG", "SHORT", or "ALL".
                           "LONG" is correct for BB/RSI (mean-reversion should
                           not enter longs in falling markets).
                           "ALL" is appropriate for MA Cross / TSMOM if used.
    exit_on_suppress     : If True, replace with EXIT when regime turns hostile
                           and position is currently open. Prevents holding a
                           stale long through a high-VIX drawdown.
    """

    def __init__(
        self,
        inner: StrategyBase,
        event_queue: EventQueue,
        data_handler: DataHandlerBase,
        vix_ticker: str = "VIX",
        vix_threshold: float = 20.0,
        vix_smoothing_window: int = 5,
        suppress_direction: str = "LONG",
        exit_on_suppress: bool = True,
    ) -> None:
        super().__init__(ticker=inner.ticker, events=event_queue, data_handler=data_handler)

        # Redirect inner strategy's queue to a staging queue so we can
        # intercept and filter before forwarding to the real event queue.
        self._staging: _stdlib_queue.Queue = _stdlib_queue.Queue()
        inner._events = self._staging
        self._inner = inner

        self._vix_ticker           = vix_ticker
        self._vix_threshold        = vix_threshold
        self._vix_smoothing_window = vix_smoothing_window
        self._suppress_direction   = suppress_direction.upper()
        self._exit_on_suppress     = exit_on_suppress
        self._in_position: bool    = False

    def _get_smoothed_vix(self) -> Optional[float]:
        """
        Rolling mean of VIX close over vix_smoothing_window bars.
        Returns None during warm-up [signals pass through unfiltered]
        """
        buf = self._data_handler.get_latest_bars(
            self._vix_ticker, N=self._vix_smoothing_window
        )
        if buf is None:
            return None
        return float(np.mean(buf["close"].cast(pl.Float64).to_numpy()))

    def _should_suppress(self, direction: str) -> bool:
        if self._suppress_direction == "ALL":
            return direction in ("LONG", "SHORT")
        return direction == self._suppress_direction

    def calculate_signals(self, event: MarketEvent) -> None:
        if event.ticker != self.ticker:
            return

        # Always run inner strategy to keep its state current
        self._inner.calculate_signals(event)

        vix = self._get_smoothed_vix()
        # If VIX warm-up not complete, pass all signals through unchanged
        risk_off: bool = (vix is not None) and (vix >= self._vix_threshold)

        exit_emitted = False
        while not self._staging.empty():
            sig = self._staging.get(block=False)

            if not isinstance(sig, SignalEvent):
                self._events.put(sig)
                continue

            if risk_off and self._should_suppress(sig.direction):
                # Risk-off: suppress signal; EXIT signal once if in position
                if self._exit_on_suppress and self._in_position and not exit_emitted:
                    self._events.put(SignalEvent(
                        ticker=self.ticker,
                        timestamp=event.timestamp,
                        direction="EXIT",
                        strength=0.0,
                    ))
                    self._in_position = False
                    exit_emitted = True
                # Signal discarded; not forwarded
            else:
                # Risk-on: forward and track position state
                if sig.direction in ("LONG", "SHORT"):
                    self._in_position = True
                elif sig.direction == "EXIT":
                    self._in_position = False
                self._events.put(sig)
