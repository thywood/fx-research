"""Triple-barrier labeling (Lopez de Prado-style), ATR-sized, for
meta-labeling a primary rule-based signal.

For each trade entry implied by a primary signal, we set:
  - an upper barrier (profit-take) and lower barrier (stop-loss), each
    `pt_mult` / `sl_mult` x ATR away from the entry price
  - a vertical barrier (time limit) of `max_holding_bars`

and record which barrier is touched first. `label` is the directional
outcome (+1 win, -1 loss, 0 flat-timeout); `meta_label` is the binary
target (1 if the primary signal's direction was profitable, 0 otherwise)
intended for a secondary ML filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def label_from_signal(
    df: pd.DataFrame,
    signal: pd.Series,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    max_holding_bars: int = 16,
    atr_window: int = 14,
) -> pd.DataFrame:
    """Compute triple-barrier labels for each new trade entry implied by
    `signal`.

    Entry convention matches `backtest.run_backtest`: position[t] =
    signal[t-1], so a new trade starting at bar `start_idx` is entered at
    `close[start_idx - 1]` (the price/ATR known when the decision was made)
    and its first bar of exposure is `start_idx`.

    Returns a DataFrame with one row per trade entry.
    """
    position = signal.shift(1).fillna(0.0)
    atr = ind.atr(df, atr_window)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    entries = (position != position.shift(1)) & (position != 0)
    entry_positions = np.where(entries.values)[0]

    n = len(df)
    rows = []
    for start_idx in entry_positions:
        if start_idx == 0:
            continue  # no prior bar to source entry price/ATR from

        side = position.iloc[start_idx]
        entry_price = close.iloc[start_idx - 1]
        atr_value = atr.iloc[start_idx - 1]
        if np.isnan(atr_value) or atr_value <= 0:
            continue

        if side > 0:
            upper, lower = entry_price + pt_mult * atr_value, entry_price - sl_mult * atr_value
        else:
            upper, lower = entry_price + sl_mult * atr_value, entry_price - pt_mult * atr_value

        end_idx = min(start_idx + max_holding_bars - 1, n - 1)

        label = None
        exit_idx = end_idx
        for i in range(start_idx, end_idx + 1):
            hit_upper = high.iloc[i] >= upper
            hit_lower = low.iloc[i] <= lower
            if hit_upper and hit_lower:
                label, exit_idx = -1, i  # ambiguous same-bar touch -> treat as loss
                break
            if hit_upper:
                label, exit_idx = (1 if side > 0 else -1), i
                break
            if hit_lower:
                label, exit_idx = (-1 if side > 0 else 1), i
                break

        if label is None:
            exit_price = close.iloc[end_idx]
            ret = (exit_price - entry_price) / entry_price * side
            label = 1 if ret > 0 else (-1 if ret < 0 else 0)
            exit_idx = end_idx

        rows.append(
            {
                "entry_time": df.index[start_idx],
                "exit_time": df.index[exit_idx],
                "side": side,
                "entry_price": entry_price,
                "atr": atr_value,
                "pt_level": upper if side > 0 else lower,
                "sl_level": lower if side > 0 else upper,
                "n_bars_held": exit_idx - start_idx + 1,
                "label": label,
                "meta_label": 1 if label == 1 else 0,
            }
        )

    return pd.DataFrame(rows)
