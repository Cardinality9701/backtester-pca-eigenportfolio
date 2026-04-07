# Backtester + PCA Eigenportfolio Creation

> Event-driven backtesting engine + Principal Component Analysis (PCA) and correlation regime exploration via eigenportfolios\
> \
> Backtesting basic strategies on SPY: Moving Average Crossover, Time Series Momentum (TSMOM), Bollinger Bands, Relative Strength Index (RSI)\
> \
> Backtesting more complex strategy: TSMOM multi-asset (SPY, TLT, GLD, EEM, DBC)\
> \
> Principal Component Analysis utilized to construct eigenportfolios for multi-asset portfolio and for only SPY

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Project Summaries

### Project 1 — PCA Eigenportfolio Regime Strategy (2008–2024)

A 4-phase pipeline that constructs rolling eigenportfolios from five multi-asset
ETFs (SPY, TLT, GLD, EEM, DBC), detects correlation regime changes via the
Absorption Ratio (ΔAR), and conditions a long-only strategy on the regime signal.

**Key finding:** Conditioning on PC2 with threshold −0.5σ lifts Sharpe by **+0.748**
on the training set (conditional Sharpe +0.506 vs. unconditional −0.242). Validation
confirms the pattern; the 2022–2024 test period is negative for all strategies
and the benchmark.

| Split | Strategy A (time exit) | Strategy B (signal exit) | Strategy C (hybrid) | Benchmark (long) |
|-------|----------------------|------------------------|-------------------|-----------------|
| Train (–2018) | +0.181 | +0.247 | +0.247 | −0.244 |
| Val (2019–2021) | +0.162 | +0.537 | +0.533 | +0.221 |
| Test (2022–2024) | −0.732 | −1.262 | −1.262 | −0.270 |

<details>
<summary>Equity curves and Sharpe-by-split figures</summary>

![Equity Curves by Split](reports/figures/fig9_equity_curves.jpg)
![Sharpe by Split](reports/figures/fig10_sr_by_split.png)

</details>

**Backtester Engine — Eigenportfolio Strategy on SPY (2008–2024):**

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 0.4365 |
| Ann. Return | 2.68% |
| Max Drawdown | −22.56% |
| Calmar Ratio | 0.12 |
| Hit Rate | 64.71% |
| Total Trades | 17 |
| Final Equity | $783,185 |

Verifiying the annual return; \
*Given initial capital $500,000; Compound Annual Growth Rate (CAGR) = ($783,185 / $500,000)^(1/17) − 1 = 2.67%*

---

### Project 2 — Classical Strategies on SPY (2005–2024)

Five rule-based strategies run through the event-driven engine on SPY with
simulated transaction costs in terms of basis points (10bps commission + 5bps adversarial slippage).

| Strategy | Period | Ann. Return | Sharpe | Max DD | Calmar | Hit Rate | Trades | Final Equity |
|----------|--------|-------------|--------|--------|--------|----------|--------|-------------|
| MA Cross | 2010–2024 | 0.21% | 0.2037 | −8.57% | 0.02 | 47.56% | 164 | $103,167 |
| TSMOM (SPY) | 2005–2024 | 0.65% | 0.4012 | −6.24% | 0.10 | 52.38% | 63 | $113,847 |
| Bollinger Band | 2005–2024 | 0.50% | 0.4882 | −5.18% | 0.10 | 50.72% | 138 | $110,392 |
| RSI | 2005–2024 | 0.04% | 0.2155 | −4.43% | 0.01 | 48.91% | 459 | $100,781 |
| TSMOM (Multi)† | 2008–2024 | 0.79% | 0.2454 | −8.51% | 0.09 | 41.12% | 518 | $571,292 |

*Initial capital $100,000 for single-asset strategies; $500,000 for TSMOM Multi.
All CAGRs independently verified: (Final Equity / Initial Capital)^(1/n_years) − 1.*

*†TSMOM Multi: SPY, TLT, GLD, EEM, DBC; position sizing each independently by the engine.*

---

## Quickstart

```bash
git clone https://github.com/Cardinality9701/backtester-pca-eigenportfolio.git
cd quant-research-platform
pip install -r requirements.txt

# Run any classical strategy
python run_backtest.py --config config/ma_cross_SPY.yaml

# Run all classical strategies
for cfg in config/ma_cross_SPY.yaml config/rsi_SPY.yaml config/bollinger_SPY.yaml \
           config/tsmom_SPY.yaml config/tsmom_multi.yaml; do
    python run_backtest.py --config "$cfg"
done

# Run the full PCA eigenportfolio pipeline
python run_eigenportfolio.py --config config/eigenportfolio.yaml

# Regenerate outputs for a strategy already run
python visualize.py --run tsmom_SPY          # equity curve + drawdown
python tearsheet.py --run ma_cross_SPY       # multi-panel tearsheet PNG
python sensitivity.py --config config/ma_cross_SPY.yaml   # parameter heatmap
python walk_forward.py --config config/tsmom_SPY.yaml     # anchored WF
```

Each run creates a timestamped directory under `outputs/` containing:
`config_snapshot.yaml` · `backtest.db` · `metrics.csv` · `tearsheet.png` · `sensitivity_heatmap.png` · `walkforward.png`

---

## Architecture

<details>
<summary><strong>Project 1 — Event-Driven Backtesting Engine</strong></summary>

The engine uses a `queue.Queue` event loop that structurally prevents look-ahead
bias. Bar N+1 is physically unavailable until bar N's entire event chain is
drained. This was done to mirror real trading infrastructure, where data events must be
processed in strict causal order.

```
DataHandler releases bar N
  → MarketEvent enters queue
  → Strategy reads ONLY bars 1..N → SignalEvent
  → RiskManager sizes position   → OrderEvent
  → ExecutionHandler fills at bar N close (+ costs) → FillEvent
  → Portfolio updates state, streams to SQLite
DataHandler releases bar N+1   ← only now
```

**Five components:**

| Component | File | Similar to |
|-----------|------|---------|
| DataHandler | `src/data_execution.py` | Bloomberg terminal feed |
| Strategy | `src/strategy.py` | Analyst generating signals |
| RiskManager | `src/portfolio_risk.py` | Risk desk sizing positions |
| ExecutionHandler | `src/data_execution.py` | Broker filling orders |
| Portfolio | `src/portfolio_risk.py` | Brokerage account ledger |

**Key design decisions:**
- **Rolling buffer:** `collections.deque(maxlen=N)` per ticker. This appends a new bar,
  removes oldest. This allowed Strategy to execute on bounded, fully-causal window
- **SQLite streaming:** equity snapshots and trade records written to DB in real
  time (minimize data stored in RAM)
- **Fixed-fractional sizing:** RiskManager risks a fixed % of current cash per
  new position (configurable via `risk_fraction`)
- **Cost model:** flat commission + adversarial slippage (BUY fills slightly
  above close, SELL slightly below)

</details>

<details>
<summary><strong>Project 2 — PCA Eigenportfolio Pipeline</strong></summary>

Research pipeline operating on SPY, TLT, GLD, EEM, DBC
(2008-01-01 to 2024-12-31, 4,278 bars).

**Eigenportfolio Construction**
Rolling exponentially weighted moving (EWM) PCA at three half-lives (21d, 63d, 126d). SVD with sign-flip
correction anchored to the first bar. \
\
Absorption Ratio = top-K eigenvalues /
total eigenvalues. Dynamic K retains 90% variance threshold. K drops to 1
during crises (2008, 2020, 2022), showing correlation concentration as
regime signal.

**Ledoit-Wolf Covariance**
Rolling shrinkage estimator (window=252). Shrinkage intensity peaks in 2022-03
and 2020-03, validating sensitivity to stress regimes.

**redictive Analysis (ΔAR → VIX)**
Tests change in absorption ratio (ΔAR) as a predictor of forward VIX for time horizons h = 1, 5, 10, 21 days.
Finding: ΔAR is a *concurrent* indicator, not a leading predictor. This
negative result motivated the regime-conditioning approach in Phase 4 over
directional forecasting.

**Phase 4 — Regime Strategy**
PC selection on train set only (no re-estimation). \
\
Selected: PC2, direction
+1, threshold −0.5σ. Three exit variants (hard time exit, signal exit, hybrid)
evaluated across train/val/test splits while avoiding look-ahead bias.

**Data splits:**
- Train: up to 2018-12-31 (2,641 days) — PC selection only
- Val: 2019-01-01 to 2021-12-31 (757 days) — OOS monitoring
- Test: 2022-01-01 to 2024-12-31 (752 days) — final OOS evaluation

</details>

---

## Repo Structure

```
backtester-pca-eigenportfolio/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                          # Event-driven backtesting engine
│   ├── events.py                 # MarketEvent, SignalEvent, OrderEvent, FillEvent
│   ├── data_execution.py         # DataHandler & ExecutionHandler
│   ├── portfolio_risk.py         # Portfolio, RiskManager, SQLite streaming
│   ├── strategy.py               # MA Cross, RSI, Bollinger, TSMOM implementations
│   ├── engine_analytics.py       # Event loop & analytics engine
│   └── grids.py                  # Parameter grids for sensitivity sweeps
│
├── eigenportfolio/               # PCA research pipeline
│   ├── data_preprocessor.py
│   ├── covariance_engine.py      # EWM PCA + Ledoit-Wolf
│   ├── spectral_decomposer.py
│   ├── risk_monitor.py           # Absorption Ratio
│   ├── eigenportfolio.py
│   ├── predictive_analysis.py    # Phase 3: ΔAR → VIX
│   ├── visualizer.py             # Figures 1–7
│   ├── regime_classifier.py      # Phase 4: PC selection, regime labelling
│   ├── signal_generator.py       # Phase 4: Exit strategies A/B/C
│   ├── backtester.py             # Phase 4: Vol-scaled backtesting
│   ├── performance_evaluator.py  # Phase 4: Performance reporting
│   └── regime_visualizer.py      # Phase 4: Figures 8–10
│
├── config/                       # One YAML per strategy run
│   ├── eigenportfolio.yaml
│   ├── ma_cross_SPY.yaml
│   ├── rsi_SPY.yaml
│   ├── bollinger_SPY.yaml
│   ├── tsmom_SPY.yaml
│   └── tsmom_multi.yaml
│
├── run_backtest.py               # Entry point: engine strategies
├── run_eigenportfolio.py         # Entry point: PCA pipeline (Phases 1–4)
├── sensitivity.py                # Parameter sweep heatmap
├── walk_forward.py               # Anchored / rolling walk-forward validation
├── tearsheet.py                  # Multi-panel tearsheet generator
├── visualize.py                  # Equity curve + drawdown plotter
│
├── outputs/                      
│   └── .gitkeep
├── data/                         
│   └── .gitkeep
│
└── reports/
    ├── eigenportfolio_memo.md
    ├── ma_cross_memo.md
    └── figures/
        ├── fig9_equity_curves.jpg
        └── fig10_sr_by_split.jpg
```

---

## Configuration

All configs for available strategies are within the config folder. Example config (MA Crossover) is shown below;

```yaml
# config/ma_cross_SPY.yaml
run_name: "ma_cross_SPY"
data:
  ticker: "SPY"
  start: "2010-01-01"
  end: "2024-12-31"
  buffer_size: 200
strategy:
  type: "MovingAverageCross"
  fast_window: 10
  slow_window: 40
risk:
  initial_capital: 100000
  risk_fraction: 0.10
execution:
  commission_pct: 0.001
  slippage_bps: 5.0
analytics:
  trading_days_per_year: 252
```

Available `strategy.type` values: `MovingAverageCross` · `RSI` · `BollingerBand` · `TSMOM` · `Eigenportfolio`

---

## Known Issues

- **`walk_forward.py` + Eigenportfolio:** raises `ValueError` because
  `src/grids.py` does not define a scalar parameter grid for the eigenportfolio
  strategy. Walk-forward sensitivity is not applicable in the same way — Phase 4
  PC selection already constitutes the model-selection step. All other strategies
  are compatible with `walk_forward.py`. Error can be safely ignored during run.

- Improve upon PCA eigenportfolio project; introduce some sort of fix (e.g. macro conditioning variable via, 
  yield curve slope, regime type classification for filtering, adaptive exiting to debug Strategies B and C)

---

## License

MIT
