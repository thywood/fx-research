"""Candlestick chart rendering for sample periods, saved to results/."""

from __future__ import annotations

import os

import mplfinance as mpf
import pandas as pd


def plot_sample(
    df: pd.DataFrame,
    start: str,
    end: str,
    out_path: str,
    title: str | None = None,
    overlays: dict | None = None,
):
    """Render a candlestick chart for df.loc[start:end] and save to out_path.

    overlays: optional dict of {label: pd.Series} aligned to df.index to plot
    as additional lines (e.g. moving averages).
    """
    window = df.loc[start:end, ["open", "high", "low", "close", "volume"]].copy()
    window.columns = ["Open", "High", "Low", "Close", "Volume"]

    addplots = []
    if overlays:
        for label, series in overlays.items():
            s = series.loc[window.index]
            addplots.append(mpf.make_addplot(s, label=label))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plot_kwargs = dict(
        type="candle",
        style="yahoo",
        title=title or f"{start} to {end}",
        volume=True,
        savefig=dict(fname=out_path, dpi=150, bbox_inches="tight"),
    )
    if addplots:
        plot_kwargs["addplot"] = addplots
    mpf.plot(window, **plot_kwargs)
