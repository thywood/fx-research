"""Render candlestick charts of sample periods for visual review.

Usage:
    .venv/bin/python scripts/02_render_charts.py data/EURUSD_M15.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fxlib import charts, data_loader  # noqa: E402

DEFAULT_CSV = "data/EURUSD_M15.csv"


def main(csv_path: str):
    df = data_loader.load_mt5_csv(csv_path)
    df = df[~df.index.duplicated(keep="first")]

    n = len(df)
    if n < 200:
        raise ValueError(f"Only {n} rows loaded -- not enough for sample charts")

    # Three sample windows: start, middle, end of the dataset (~5 trading
    # days / ~480 M15 bars each).
    window_bars = 480
    samples = {
        "start": df.iloc[0:window_bars],
        "middle": df.iloc[n // 2 : n // 2 + window_bars],
        "end": df.iloc[-window_bars:],
    }

    for name, window in samples.items():
        out_path = f"results/sample_{name}_{window.index[0].date()}_{window.index[-1].date()}.png"
        charts.plot_sample(
            df,
            start=str(window.index[0]),
            end=str(window.index[-1]),
            out_path=out_path,
            title=f"EUR/USD M15 - {name} sample ({window.index[0].date()} to {window.index[-1].date()})",
        )
        print(f"Saved {out_path}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    main(csv_path)
