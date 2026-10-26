"""Rolling-window walk-forward validation.

Designed so a rule-based baseline (fixed parameters, `fit` is a no-op) and
a future ML meta-labeling model (which would `fit` a classifier on the
train window's triple-barrier labels and use it to filter/size the primary
signal on the test window) share the same interface:

    class Strategy:
        def fit(self, train_df: pd.DataFrame) -> None: ...
        def signal(self, context_df: pd.DataFrame) -> pd.Series: ...

`signal()` is called on train+test combined ("context") so indicators have
enough lookback at the start of each test window; only the test portion of
the resulting signal/position/returns is used for that window's reported
metrics.
"""

from __future__ import annotations

import pandas as pd

from . import backtest as bt
from . import costs as cost_mod


class RuleBasedStrategy:
    """Adapter for a fixed-parameter rule-based signal function. `fit` is a
    no-op -- present only so this matches the interface a future ML
    meta-labeling strategy would implement."""

    def __init__(self, signal_fn, **params):
        self.signal_fn = signal_fn
        self.params = params

    def fit(self, train_df: pd.DataFrame) -> None:
        pass

    def signal(self, context_df: pd.DataFrame) -> pd.Series:
        return self.signal_fn(context_df, **self.params)


def generate_windows(n: int, train_size: int, test_size: int, step_size: int) -> list[tuple[int, int, int, int]]:
    """Return list of (train_start, train_end, test_start, test_end) index
    positions for rolling train/test windows."""
    windows = []
    train_start = 0
    while True:
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + test_size
        if test_end > n:
            break
        windows.append((train_start, train_end, test_start, test_end))
        train_start += step_size
    return windows


def walk_forward(
    df: pd.DataFrame,
    strategy_factory,
    train_size: int,
    test_size: int,
    step_size: int,
    round_turn_pips: float = cost_mod.DEFAULT_ROUND_TURN_PIPS,
    pip_size: float = cost_mod.PIP_SIZE_EURUSD,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Run rolling walk-forward validation.

    Returns:
      window_metrics: one row per window with metrics + window date ranges
      oos_net_returns: concatenated out-of-sample net returns across all
        windows (continuous, for an overall OOS equity curve)
      overall_metrics: metrics computed over the full concatenated OOS series
    """
    n = len(df)
    windows = generate_windows(n, train_size, test_size, step_size)
    if not windows:
        raise ValueError(
            f"No windows fit: n={n}, train_size={train_size}, test_size={test_size}, step_size={step_size}"
        )

    window_rows = []
    oos_returns_parts = []
    oos_trades_parts = []

    for train_start, train_end, test_start, test_end in windows:
        train_df = df.iloc[train_start:train_end]
        context_df = df.iloc[train_start:test_end]
        test_index = df.index[test_start:test_end]

        strategy = strategy_factory()
        strategy.fit(train_df)
        full_signal = strategy.signal(context_df)

        result, trades, _ = bt.run_backtest(context_df, full_signal, round_turn_pips, pip_size)

        net_returns_slice = result.loc[test_index, "net_return"]
        equity_slice = (1 + net_returns_slice).cumprod()

        if not trades.empty:
            trades_in_window = trades[(trades["start"] >= test_index[0]) & (trades["start"] <= test_index[-1])]
        else:
            trades_in_window = trades

        metrics = bt.compute_metrics(net_returns_slice, equity_slice, trades_in_window)
        metrics["train_start"] = df.index[train_start]
        metrics["train_end"] = df.index[train_end - 1]
        metrics["test_start"] = df.index[test_start]
        metrics["test_end"] = df.index[test_end - 1]

        window_rows.append(metrics)
        oos_returns_parts.append(net_returns_slice)
        oos_trades_parts.append(trades_in_window)

    oos_net_returns = pd.concat(oos_returns_parts)
    oos_equity = (1 + oos_net_returns).cumprod()
    oos_trades = pd.concat(oos_trades_parts, ignore_index=True) if oos_trades_parts else pd.DataFrame()
    overall_metrics = bt.compute_metrics(oos_net_returns, oos_equity, oos_trades)

    window_metrics = pd.DataFrame(window_rows)
    cols = ["train_start", "train_end", "test_start", "test_end"] + [
        c for c in window_metrics.columns if c not in ("train_start", "train_end", "test_start", "test_end")
    ]
    window_metrics = window_metrics[cols]

    return window_metrics, oos_net_returns, overall_metrics
