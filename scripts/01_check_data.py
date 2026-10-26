"""Load an MT5 OHLCV CSV export and run data-quality checks.

Usage:
    .venv/bin/python scripts/01_check_data.py data/EURUSD_M15.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fxlib import data_loader, quality  # noqa: E402

DEFAULT_CSV = "data/EURUSD_M15.csv"


def main(csv_path: str):
    df = data_loader.load_mt5_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Date range: {df.index.min()} -> {df.index.max()}")
    print(df.head())
    print()

    report = quality.summarize(df)

    print("=== Data quality summary ===")
    print(f"Rows:                {report['n_rows']}")
    print(f"Date range:          {report['date_range'][0]} -> {report['date_range'][1]}")
    print(f"Duplicate timestamps:{report['n_duplicates']}")
    print(f"Bad OHLC rows:       {report['n_bad_ohlc']}")
    print(f"Total gaps:          {report['n_gaps_total']}")
    print(f"  of which unexpected (not weekend): {report['n_gaps_unexpected']}")
    print(f"Outlier bars:        {report['n_outliers']}")
    print(f"Flat/stale runs:     {report['n_flat_runs']}")

    if report["n_duplicates"] > 0:
        print("\n--- Duplicate timestamps (first 10) ---")
        print(report["duplicates"].head(10))

    if report["n_bad_ohlc"] > 0:
        print("\n--- Bad OHLC rows (first 10) ---")
        print(report["bad_ohlc"].head(10))

    if report["n_gaps_unexpected"] > 0:
        print("\n--- Unexpected (non-weekend) gaps (first 10) ---")
        print(report["unexpected_gaps"].head(10))

    if report["n_outliers"] > 0:
        print("\n--- Outlier bars (first 10) ---")
        print(report["outliers"].head(10)[["open", "high", "low", "close", "ret", "ret_z", "range_z"]])

    if report["n_flat_runs"] > 0:
        print("\n--- Flat/stale runs (first 10) ---")
        print(report["flat_runs"].head(10))

    out_path = Path("results") / "data_quality_summary.txt"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"CSV: {csv_path}\n")
        f.write(f"Rows: {report['n_rows']}\n")
        f.write(f"Date range: {report['date_range'][0]} -> {report['date_range'][1]}\n")
        f.write(f"Duplicate timestamps: {report['n_duplicates']}\n")
        f.write(f"Bad OHLC rows: {report['n_bad_ohlc']}\n")
        f.write(f"Total gaps: {report['n_gaps_total']}\n")
        f.write(f"Unexpected (non-weekend) gaps: {report['n_gaps_unexpected']}\n")
        f.write(f"Outlier bars: {report['n_outliers']}\n")
        f.write(f"Flat/stale runs: {report['n_flat_runs']}\n")
        if report["n_gaps_unexpected"] > 0:
            f.write("\nUnexpected gaps:\n")
            f.write(report["unexpected_gaps"].to_string())
            f.write("\n")
        if report["n_outliers"] > 0:
            f.write("\nOutlier bars:\n")
            f.write(report["outliers"][["open", "high", "low", "close", "ret", "ret_z", "range_z"]].to_string())
            f.write("\n")
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    main(csv_path)
