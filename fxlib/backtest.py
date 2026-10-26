"""Vectorized single-asset backtest with transaction costs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import costs as cost_mod

# Approximate M15 bars/year for forex (24h x 5 days x 4 bars/hour x 52 weeks).
# Used only for annualizing return/vol/Sharpe -- a rough constant is fine
# since it's applied consistently across strategies/windows.
BARS_PER_YEAR_M15 = 24960


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    round_turn_pips: float = cost_mod.DEFAULT_ROUND_TURN_PIPS,
    pip_size: float = cost_mod.PIP_SIZE_EURUSD,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run a backtest of `signal` (target position at each bar, computed
    from data available at that bar's close) against `df`.

    The signal is shifted by one bar before being applied as a position,
    i.e. a decision made at the close of bar t is acted on during bar t+1.

    Returns (result_df, trades_df, metrics_dict).
    """
    close = df["close"]
    position = signal.shift(1).fillna(0.0)

    price_return = close.pct_change().fillna(0.0)
    gross_return = position * price_return

    position_change = position.diff()
    position_change.iloc[0] = position.iloc[0]  # entering from flat at the start
    cost_per_unit = cost_mod.cost_per_unit_position_change(
        close.shift(1), round_turn_pips, pip_size
    ).fillna(0.0)
    cost = position_change.abs() * cost_per_unit

    net_return = gross_return - cost
    equity = (1 + net_return).cumprod()

    result = pd.DataFrame(
        {
            "position": position,
            "gross_return": gross_return,
            "cost": cost,
            "net_return": net_return,
            "equity": equity,
        }
    )

    trades = extract_trades(position, net_return)
    metrics = compute_metrics(net_return, equity, trades)
    return result, trades, metrics


def extract_trades(position: pd.Series, net_return: pd.Series) -> pd.DataFrame:
    """Group consecutive bars with a constant non-zero position into trades
    and compute each trade's compounded net return."""
    trades = []
    current_pos = 0.0
    trade_returns: list[float] = []
    trade_start = None
    last_ts = None

    for ts, pos, ret in zip(position.index, position.values, net_return.values):
        if pos != current_pos:
            if current_pos != 0.0 and trade_returns:
                trades.append(
                    {
                        "start": trade_start,
                        "end": last_ts,
                        "direction": current_pos,
                        "n_bars": len(trade_returns),
                        "return": _compound(trade_returns),
                    }
                )
            if pos != 0.0:
                trade_start = ts
                trade_returns = []
            current_pos = pos
        if pos != 0.0:
            trade_returns.append(ret)
        last_ts = ts

    if current_pos != 0.0 and trade_returns:
        trades.append(
            {
                "start": trade_start,
                "end": last_ts,
                "direction": current_pos,
                "n_bars": len(trade_returns),
                "return": _compound(trade_returns),
            }
        )

    return pd.DataFrame(trades)


def _compound(returns: list[float]) -> float:
    out = 1.0
    for r in returns:
        out *= 1 + r
    return out - 1.0


def compute_metrics(
    net_return: pd.Series, equity: pd.Series, trades: pd.DataFrame, bars_per_year: int = BARS_PER_YEAR_M15
) -> dict:
    n = len(net_return)
    total_return = float(equity.iloc[-1] - 1.0) if n > 0 else np.nan
    mean_r = net_return.mean()
    std_r = net_return.std()
    sharpe = float(mean_r / std_r * np.sqrt(bars_per_year)) if std_r > 0 else np.nan
    ann_return = float((1 + total_return) ** (bars_per_year / n) - 1) if n > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = float(drawdown.min()) if n > 0 else np.nan

    n_trades = len(trades)
    if n_trades > 0:
        win_rate = float((trades["return"] > 0).mean())
        gains = trades.loc[trades["return"] > 0, "return"].sum()
        losses = trades.loc[trades["return"] < 0, "return"].sum()
        profit_factor = float(gains / abs(losses)) if losses != 0 else np.inf
        avg_trade_return = float(trades["return"].mean())
        avg_trade_bars = float(trades["n_bars"].mean())
    else:
        win_rate = np.nan
        profit_factor = np.nan
        avg_trade_return = np.nan
        avg_trade_bars = np.nan

    return {
        "n_bars": n,
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": float(std_r * np.sqrt(bars_per_year)),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_return": avg_trade_return,
        "avg_trade_bars": avg_trade_bars,
    }
