"""Backtest the baseline rule-based signal generators with transaction costs.

Usage:
    .venv/bin/python scripts/03_backtest_baselines.py data/EURUSD_M15.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fxlib import backtest as bt  # noqa: E402
from fxlib import costs as cost_mod  # noqa: E402
from fxlib import data_loader, signals  # noqa: E402

DEFAULT_CSV = "data/EURUSD_M15.csv"

BASELINE_PARAMS = {
    "ma_crossover": dict(fast=10, slow=50),
    "rsi_mean_reversion": dict(rsi_window=14, oversold=30, overbought=70, exit_mid=50),
    "atr_breakout": dict(channel_window=20, atr_window=14, atr_mult=0.5),
}


def main(csv_path: str):
    df = data_loader.load_mt5_csv(csv_path)
    df = df[~df.index.duplicated(keep="first")]

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(
        f"Transaction cost: {cost_mod.DEFAULT_ROUND_TURN_PIPS} pip round-turn "
        f"(PLACEHOLDER -- confirm vs BlackBull's actual EUR/USD spread+commission)\n"
    )

    rows = []
    for name, fn in signals.BASELINES.items():
        params = BASELINE_PARAMS[name]
        signal = fn(df, **params)
        _, trades, metrics = bt.run_backtest(df, signal)
        row = {"strategy": name, "params": str(params), **metrics}
        rows.append(row)

        print(f"=== {name} {params} ===")
        for k, v in metrics.items():
            print(f"  {k:20s}: {v}")
        print()

    summary = pd.DataFrame(rows)
    out_path = Path("results") / "baseline_backtest_summary.csv"
    out_path.parent.mkdir(exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    main(csv_path)
