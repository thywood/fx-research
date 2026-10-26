"""Baseline rule-based signal generators.

Each function returns a pd.Series of target positions aligned to df.index:
  +1 = long, -1 = short, 0 = flat.

All signals are computed from information available *at* bar t (using only
past data via .shift()/.rolling() conventions); the backtester is
responsible for shifting the signal by one bar before applying it as a
position, so there's no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def ma_crossover_signal(df: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.Series:
    """Trend-following: long when fast EMA > slow EMA, short when below."""
    fast_ma = ind.ema(df["close"], fast)
    slow_ma = ind.ema(df["close"], slow)

    signal = pd.Series(0.0, index=df.index, name="signal")
    signal[fast_ma > slow_ma] = 1.0
    signal[fast_ma < slow_ma] = -1.0
    signal[fast_ma.isna() | slow_ma.isna()] = 0.0
    return signal


def rsi_mean_reversion_signal(
    df: pd.DataFrame,
    rsi_window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    exit_mid: float = 50.0,
) -> pd.Series:
    """Counter-trend: go long when RSI dips below `oversold`, go short when
    it rises above `overbought`. Exit to flat when RSI crosses back through
    `exit_mid`. Stateful (simple state machine over RSI values)."""
    rsi = ind.rsi(df["close"], rsi_window).values

    pos = np.zeros(len(df))
    current = 0.0
    for i, r in enumerate(rsi):
        if np.isnan(r):
            pos[i] = 0.0
            continue
        if current == 0.0:
            if r < oversold:
                current = 1.0
            elif r > overbought:
                current = -1.0
        elif current == 1.0:
            if r >= exit_mid:
                current = 0.0
        elif current == -1.0:
            if r <= exit_mid:
                current = 0.0
        pos[i] = current

    return pd.Series(pos, index=df.index, name="signal")


def atr_breakout_signal(
    df: pd.DataFrame,
    channel_window: int = 20,
    atr_window: int = 14,
    atr_mult: float = 0.5,
) -> pd.Series:
    """Trend-following breakout: go long when close breaks above the prior
    `channel_window`-bar high plus `atr_mult` * ATR, go short on the
    symmetric break below the prior low. Holds position until the opposite
    breakout fires (stateful)."""
    atr = ind.atr(df, atr_window)
    rolling_high = df["high"].rolling(channel_window).max().shift(1)
    rolling_low = df["low"].rolling(channel_window).min().shift(1)

    upper = (rolling_high + atr_mult * atr).values
    lower = (rolling_low - atr_mult * atr).values
    close = df["close"].values

    pos = np.zeros(len(df))
    current = 0.0
    for i in range(len(df)):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            pos[i] = current
            continue
        if close[i] > upper[i]:
            current = 1.0
        elif close[i] < lower[i]:
            current = -1.0
        pos[i] = current

    return pd.Series(pos, index=df.index, name="signal")


BASELINES = {
    "ma_crossover": ma_crossover_signal,
    "rsi_mean_reversion": rsi_mean_reversion_signal,
    "atr_breakout": atr_breakout_signal,
}
