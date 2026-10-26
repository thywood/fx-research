# fx-research

A small Python library for testing foreign-exchange trading rules honestly: clean the data first, charge realistic costs on every trade, and check whether a rule survives out of sample before believing it.

It works on 15-minute OHLCV candle exports from MetaTrader 5. The repo ships with a synthetic EUR/USD sample (with a few deliberate gaps, duplicates and outliers) so everything runs without real market data.

## What it does

| Module | Purpose |
|---|---|
| `fxlib/data_loader.py` | Load and clean MT5 History Center CSV exports |
| `fxlib/quality.py` | Find duplicate timestamps, bad OHLC rows, non-weekend gaps, outlier bars and stale runs |
| `fxlib/charts.py` | Render candlestick charts of sample periods for visual review |
| `fxlib/indicators.py` | Moving averages, RSI and ATR |
| `fxlib/signals.py` | Baseline rules: MA crossover, RSI mean reversion, ATR channel breakout |
| `fxlib/costs.py` | Spread and commission charged on every position change |
| `fxlib/backtest.py` | Vectorised single-asset backtest with return, volatility, Sharpe, drawdown and trade stats |
| `fxlib/labeling.py` | ATR-sized triple-barrier labelling of trade entries |
| `fxlib/walkforward.py` | Rolling walk-forward windows for out-of-sample checks |

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/01_check_data.py data/_synthetic_EURUSD_M15.csv
.venv/bin/python scripts/02_render_charts.py data/_synthetic_EURUSD_M15.csv
.venv/bin/python scripts/03_backtest_baselines.py data/_synthetic_EURUSD_M15.csv
.venv/bin/python scripts/04_triple_barrier.py data/_synthetic_EURUSD_M15.csv
```

Outputs land in `results/`: a data-quality report, sample charts and a backtest summary.

## A note on results

The numbers in `results/` come from synthetic data and say nothing about real markets. The point of the project is the pipeline: data checks, cost-aware backtesting and out-of-sample validation. It is not a trading strategy and not financial advice.

## Stack

Python, pandas, NumPy, Matplotlib, mplfinance, SciPy, scikit-learn.
