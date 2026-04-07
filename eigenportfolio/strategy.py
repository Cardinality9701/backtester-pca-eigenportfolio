
from __future__ import annotations

import queue
import numpy as np
from abc import abstractmethod
from typing import Dict, List, Tuple, Optional
import polars as pl

from src.strategy import PortfolioStrategy


class EigenportfolioStrategy(PortfolioStrategy):
    """
    PCA-based eigenportfolio strategy for the event-driven engine.

    On each rebalance bar:
      1. Pull price matrix from data handler buffer
      2. Compute log returns
      3. Estimate sample covariance matrix
      4. Eigendecompose → eigenvectors sorted by descending eigenvalue
      5. Use first eigenvector (max-variance direction) as raw weights
      6. Vol-scale so realised portfolio vol targets vol_target
      7. Release one SignalEvent per ticker with weight as strength

    Self-contained — uses numpy linalg directly, no dependency on
    other eigenportfolio module files. Those files are used by
    run_eigenportfolio.py for the standalone PCA research pipeline.
    """

    def __init__(
        self,
        tickers: List[str],
        event_queue: queue.Queue,
        data_handler,
        lookback: int        = 252,
        n_components: int    = 1,
        rebalance_every: int = 21,
        vol_target: float    = 0.10,
    ) -> None:
        super().__init__(tickers, event_queue, data_handler, rebalance_every)
        self.lookback       = lookback
        self.n_components   = n_components
        self.vol_target     = vol_target

    def compute_weights(
        self,
        price_matrix: np.ndarray,
        tickers: List[str],
    ) -> Dict[str, float]:
        """
        price_matrix: shape (n_bars, n_tickers) — close prices
        Returns target weights keyed by ticker.
        """
        n_tickers = len(tickers)
        prices = price_matrix[-self.lookback:]

        if prices.shape[0] < 30 or prices.shape[1] != n_tickers:
            return {t: 0.0 for t in tickers}

        # Log returns — shape (T-1, N)
        returns = np.diff(np.log(prices), axis=0)

        # Drop any columns with NaN/inf (data gaps)
        valid_mask = np.all(np.isfinite(returns), axis=0)
        if valid_mask.sum() < 2:
            return {t: 0.0 for t in tickers}

        ret_clean = returns[:, valid_mask]
        tickers_clean = [t for t, v in zip(tickers, valid_mask) if v]

        # Sample covariance (annualised)
        cov = np.cov(ret_clean.T) * 252

        # Eigendecomposition — eigh returns ascending eigenvalues
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort descending — largest variance first
        idx  = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        # First eigenvector = max-variance portfolio
        w = eigvecs[:, 0].astype(float)

        # Sign convention — align so majority of weights are positive
        if np.sum(w) < 0:
            w = -w

        # Vol-scale to target annualised portfolio volatility
        port_vol = float(np.sqrt(w @ cov @ w))
        if port_vol > 1e-8:
            w = w * (self.vol_target / port_vol)

        # Clip each weight to [-1, 1]
        w = np.clip(w, -1.0, 1.0)

        # Map back to full ticker list (zeroing out any dropped tickers)
        weight_map = dict(zip(tickers_clean, w))
        return {t: float(weight_map.get(t, 0.0)) for t in tickers}
