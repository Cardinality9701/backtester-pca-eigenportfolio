# Project Explainer: De-jargoning the Repo
 
This document explains the concepts, design decisions, and results of this 
project in plain terms, with the assumption that the reader doesn't have a financial background.
 
---
 
## Introduction
 
This project serves as the start of a research platform for testing and designing/studying stock market trading strategies. It has two parts:
a **backtesting engine** and a **regime detection pipeline**.
 
Backtesting your trading strategy is the practice of evaluating how well it performs on historical returns for whatever stock/index you're interested in. The catch with backtesting is that it is easy to accidentally cheat. If your simulation uses information that wouldn't have been available on the day of the trade (e.g. future price to inform today's trading decision), the strategy's performance is artificially higher than it should be. This is termed as **look-ahead bias**, and it is one of the most common mistakes in quantitative finance. To avoid this, the backtesting engine was designed to only see data that existed up to and including that day.
 
The second part of the project asks a deeper question: *"Can we detect what kind of market environment we're in, and use that to make smarter decisions?"* 

Markets don't behave the same way all the time, with periods of calm and steady growth behaving very differently from periods of panic and sharp selloffs. The regime
detection pipeline uses a mathematical technique called PCA (explained below) to identify what kind of environment the market is currently in, and conditions trading decisions accordingly.
 
---
 
## The Two Projects
 
### Project 1 — Classical Rule-Based Strategies (2005–2024)
 
Four well-known trading rules were implemented and tested on SPY, with one of them also being tested for a multi-asset position. Each one is set up as a systematic rule that is automatically executed.
 
**Moving Average Crossover**
 
A moving average smooths out day-to-day price noise by averaging prices over a rolling window. A crossover strategy buys when a short-term average (e.g. 50-day) rises above a long-term average (e.g. 200-day). This signals that recent momentum is building. Conversely, it sells when the opposite occurs. It is one of the  oldest and most widely studied rules in technical analysis.
 
**RSI (Relative Strength Index)**
 
RSI measures how fast and how much a price has moved recently, producing a number between 0 and 100. A reading above 70 is interpreted as "overbought" (the price has risen too far and too fast; will likely reverse), and below 30 as "oversold" (the opposite). This strategy bets against extreme recent moves.
 
**Bollinger Bands**
 
Bollinger Bands place an upper and lower boundary around a moving average, set at a fixed number of standard deviations above and below it. When price touches the upper band, the asset is considered expensive relative to recent history. Likewise, touching the lower band signals it may be cheap. This strategy trades on the concept of mean reversion (i.e. tendency of prices to drift back toward their average).
 
**Time-Series Momentum (TSMOM)**
 
Momentum is the empirical observation that assets which have risen over the past 6–12 months tend to continue rising over the next month, and vice versa. TSMOM applies this rule to a single asset (SPY) and across a universe of assets simultaneously (the multi-asset version), going long (i.e. expect increase) on recent winners and short (i.e. expect decrease) or flat (no change) on recent losers.
 
---
 
### Project 2 — Regime Detection with PCA (2008–2024)
 
**What is PCA?**
 
PCA (Principal Component Analysis) is a technique for finding hidden patterns in a large set of numbers. Imagine you're tracking a basket of different assets; in the simulated case here, US stocks, bonds, gold, emerging market stocks and commodities (SPY, TLT, GLD, EEM, DBC) were used. Each one moves up and down every day. The goal of PCA here is to find if there is some kind of underlying explanation/correlation for these assets. The answer is usually yes, with the first principal component typically capturing the broad market-wide factor that lifts or sinks most assets together.
 
**What is an Eigenportfolio?**
 
An eigenportfolio is a synthetic portfolio constructed from the output of PCA. Rather than holding SPY or TLT directly, you hold a weighted combination of all five assets in proportions determined by PCA. Each eigenportfolio captures a different "theme". For example, the first captures overall market direction, the second might capture a stocks-vs-bonds rotation, and so on. Tracking these themes over time reveals structural shifts in how assets relate to each other.
 
**What is a Market Regime?**
 
A regime is simply a label for the kind of market environment that currently exists. An example might be "low volatility and trending upward" vs. "high volatility and correlated selloff." This project uses a measure called the Absorption Ratio (ΔAR), which attempts to track how much of the market's total movement is being explained by a small number of eigenportfolios. This is then used to detect when the regime has shifted.
 
**What did we find?**
 
Conditioning trades on the regime signal (specifically, the second principal component crossing a threshold) improved the Sharpe Ratio by +0.748 on the training set, and the validation period showed the pattern was not just a coincidence. However, the strategy deteriorated in the 2022–2024 test period; the reasons for this are explored near the end of the explainer.
 
---
 
## The Engine: How the Backtest Works
 
### The Event Loop
 
The backtesting engine simulates trading by replaying historical market data one day at a time, in sequence. On each day, the engine executes a fixed sequence of steps in strict order:

1\. **Market opens**: the engine receives that day's price data
 
2\. **Signals are generated**: the strategy looks at historical data up to today and decides whether to buy, sell, or hold
 
3\. **Orders are placed**: those decisions are queued as pending orders
 
4\. **Orders are filled**: pending orders are executed at that day's closing price
 
5\. **Portfolio is updated**: cash, positions, and performance metrics are recorded
 
6\. **Market closes**: the engine moves to the next day and repeats
 
Note that the signal on step 2 can only see data from step 1 of the  *current*  day and all prior days. It cannot see the fill price from step 4 on the same day. This ordering is what makes look-ahead bias structurally impossible. This was a constraint built into the backtesting architecture.
 
### Why SQLite?
 
Every run of the engine saves its results to a SQLite database file. SQLite is a lightweight database. Think of it as a structured spreadsheet that lives in a single file and can be queried precisely. This matters for two reasons. First, it makes runs fully reproducible, with every trade, signal, and metric stored, timestamped and annotated with the exact configuration that produced it. Second, it enables comparison across runs. You can query two different strategy configurations and compare their results row by row, instead of manually analyzing two separate CSV files.
 
### Timestamped Output Directories
 
Each run creates a new folder under `outputs/` named with a timestamp (e.g. `outputs/ma_cross_SPY_20240312_143022/`). This structure means you can always trace any result back to the exact configuration that produced it. Inside it you will always find:
 
\- `config_snapshot.yaml`: the exact settings used for that run
 
\- `backtest.db`: the full trade-by-trade SQLite record
 
\- `metrics.csv`: the summary performance statistics
 
\- `tearsheet.png`: a multi-panel visual summary of the strategy's performance
 
---
 
## Key Numbers: What They Actually Mean
 
Every backtest produces a set of performance metrics. Below is an explanation of the meaning/significance of each one.
 
**Sharpe Ratio**: Return earned per unit of risk taken. A higher number means you are being better compensated for the volatility (price fluctuations) you are sitting through.
- Above 1.0 is considered good; above 2.0 is exceptional. The strategies here range from 0.20 to 0.49.
 
**Max Drawdown**: The worst peak-to-trough loss you would have experienced if you were holding throughout. If your portfolio grew from $100 to $120 then fell to $90, your max drawdown is −25%.
- Closer to 0% is better. A large drawdown means a long, painful recovery period.
 
**Calmar Ratio**: Annualized return divided by the absolute value of max drawdown. It asks: "How much return did I get per unit of worst-case pain?"
- Higher is better. A Calmar of 1.0 means you earned back your worst drawdown in one year.
 
**Hit Rate**: The percentage of individual trades that were profitable.
- 50% is the coin-flip baseline. Higher is better, but a strategy with a low hit rate can still be profitable if its winning trades are much larger than its losers.
 
**CAGR**: Compound Annual Growth Rate: the steady annual growth rate that would produce the same final result as the strategy.
- Higher is better. It is a cleaner measure of return than total return because it accounts for the length of the backtest. 
 
**Final Equity**: The value of the portfolio at the end of the backtest period, starting from the initial capital.
- Only meaningful relative to the starting capital and the time period.
 
### A Note on Project Results
 
The strategies in this project produce modest Sharpe Ratios (0.20–0.49) and low CAGRs, which was to be expected. Simple rule-based strategies applied to a highly liquid, widely studied asset like SPY have been arbitraged down over decades of academic and institutional attention. The value of designing the implementation was to build and validate a research infrastructure that could eventually apply more nuanced trading strategies to less efficient markets or more sophisticated signals.
 
---
 
## What Worked, What Didn't, and Why
 
### The Classical Strategies
The four rule-based strategies produced modest but positive results on SPY over the full backtest period. Among them, Bollinger Bands had the highest Sharpe
Ratio (0.49), and TSMOM had the highest CAGR (0.65% annualized on SPY). RSI produced the weakest results despite generating the most trades (459), suggesting
that overbought/oversold signals on a broad index like SPY have little predictive power over short horizons.
 
The multi-asset TSMOM strategy produced the highest final equity ($571,292 from $500,000) but at the cost of a higher max drawdown (−8.51%) and a relatively low hit rate (41.12%). This highlights a common tradeoff in momentum strategies: they tend to win big when they're right at the cost of taking many small losses in choppy, trendless markets.
 
None of the strategies consistently beat a simple buy-and-hold position in SPY over the full period, reflecting how competitive and well-studied these rules are in one of the world's most efficient markets.
 
### The PCA Eigenportfolio Strategy
 
**What worked:** The regime conditioning signal showed promise in the training and validation periods. Conditioning on the second principal component
(PC2) crossing −0.5 standard deviations lifted the Sharpe Ratio by +0.748 on the training set, and the validation period showed the pattern was not just a coincidence.
 
**What didn't:** The strategy produced negative Sharpe Ratios for all three variants in the 2022–2024 test period (−0.732 to −1.262), underperforming even the benchmark.
 
**Why did this happen?** One possible explanation is that the 2022–2024 period was a structural anomaly. With the Federal Reserve raising interest rates at the
fastest pace in four decades, the historical correlation structure between assets that PCA had learned from was broken. Bonds (TLT) and stocks (SPY) fell
simultaneously, which directly undermines the regime patterns the model was trained on. In other words, the model learned patterns from a world where stocks
and bonds moved in opposite directions, and then was tested in an environment where the correlation regime was something it had not experienced.
 
**What this means for the project:** It demonstrates that regime-based strategies 
are sensitive to structural breaks in correlation, which is a known challenge in 
quantitative finance called non-stationarity. Detecting and adapting to these 
breaks (rather than assuming past correlations persist) is an active area of 
research, and is a possible extension worth exploring should this project be 
expanded/developed further.
 
---
 
## Glossary
 
**Absorption Ratio (ΔAR)**
 
A measure of how much of the total variance across a set of assets is explained 
by a small number of principal components. A rising absorption ratio suggests 
markets are becoming more correlated; often a warning sign of systemic stress.
 
**Backtesting**
 
The practice of evaluating a trading strategy on historical data to estimate how 
it would have performed in the past.
 
**Benchmark**
 
A standard used for comparison. In this project, the benchmark is a simple 
long-only position in SPY (i.e. buying and holding the S&P 500 ETF throughout 
the entire period).
 
**Bollinger Bands**
 
A technical indicator that places upper and lower boundaries around a moving 
average, calculated using standard deviations. Used here to identify when an 
asset's price has moved unusually far from its recent average.
 
**CAGR (Compound Annual Growth Rate)**
 
The steady annual growth rate that would produce the same final portfolio value 
as the strategy. Accounts for the effect of compounding over time.
 
**Calmar Ratio**
 
Annualised return divided by the absolute value of maximum drawdown. Measures 
how much return was earned per unit of worst-case loss.
 
**Drawdown**
 
The percentage decline from a portfolio's peak value to its subsequent trough. 
Maximum drawdown is the largest such decline over the entire backtest period.
 
**Eigenportfolio**
 
A synthetic portfolio whose weights are determined by PCA. Each eigenportfolio 
captures a distinct statistical "theme" in asset co-movement.
 
**Event Loop**
 
The core mechanism of the backtesting engine. It processes market data one day 
at a time in a fixed sequence:
 
Receive data → generate signal → place order → fill order → update portfolio.
 
**Hit Rate**
 
The proportion of individual trades that were profitable. A hit rate of 50% means 
half of all trades made money.
 
**Look-ahead Bias**
 
The error of using information in a backtest that would not have been available at 
the time of the simulated trade. Produces artificially inflated results if present 
and makes strategies look as if they are performing better than they really are.
 
**Moving Average**
 
The average of an asset's price over a rolling window of past days. Smooths out 
short-term noise to reveal the underlying trend.
 
**Momentum**
 
The empirical tendency for assets that have performed well recently to continue 
performing well in the near future, and vice versa.
 
**Non-stationarity**
 
The property of a statistical relationship that changes over time. A non-stationary 
pattern learned from historical data may not hold in the future if the underlying 
market structure shifts.
 
**PCA (Principal Component Analysis)**
 
A mathematical technique that identifies the most important underlying factors 
driving variation in a dataset. In finance, it is used to find the dominant forces 
moving a group of assets.
 
**Principal Component (PC)**
 
An underlying factor identified by PCA. The components are typically ordered by how much of the variance in the data
they capture. PC1 captures the most variance, PC2 the second most, and so on. 
Each is meant to represent a different statistical "theme" in the data.
 
**Regime**
 
A label for the current market environment, characterised by a distinct combination 
of volatility, correlation, and trend behaviour.
 
**RSI (Relative Strength Index)**
 
A momentum indicator that measures the speed and magnitude of recent price changes, 
producing a value between 0 and 100. Values above 70 suggest an asset is overbought; 
below 30 suggests oversold.
 
**Sharpe Ratio**
 
A measure of risk-adjusted return. Calculated as the average excess return (above 
the risk-free rate) divided by the standard deviation of returns. Higher values 
indicate better compensation for the risk taken.
 
**Standard Deviation**
 
A statistical measure of how much a set of values varies around its mean. In 
finance, it is used as a proxy for risk or volatility.
 
**TSMOM (Time-Series Momentum)**
 
A momentum strategy that goes long on an asset if its recent return is positive and 
flat or short if negative. Applied here both to a single asset (SPY) and across a 
multi-asset portfolio.
