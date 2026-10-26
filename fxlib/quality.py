"""Data quality checks for M15 OHLCV forex data."""

from __future__ import annotations

import pandas as pd


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose timestamp index is duplicated."""
    mask = df.index.duplicated(keep=False)
    return df.loc[mask]


def check_ohlc_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where high/low/open/close violate basic OHLC ordering,
    or where any price is non-positive."""
    bad = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    return df.loc[bad]


def check_gaps(df: pd.DataFrame, expected_minutes: int = 15) -> pd.DataFrame:
    """Classify time gaps between consecutive bars.

    Returns a DataFrame (one row per gap > expected) with columns:
    gap_start, gap_end, gap_minutes, spans_weekend
    """
    idx = df.index
    deltas = idx.to_series().diff().dropna()
    expected = pd.Timedelta(minutes=expected_minutes)
    gaps = deltas[deltas > expected]

    rows = []
    for end_ts, delta in gaps.items():
        start_ts = end_ts - delta
        # A weekend gap is one where the interval [start_ts, end_ts] includes a Saturday,
        # which is the normal closed period for forex (Fri evening -> Sun evening).
        span_days = pd.date_range(start_ts, end_ts, freq="D")
        spans_weekend = any(d.weekday() == 5 for d in span_days)
        rows.append(
            {
                "gap_start": start_ts,
                "gap_end": end_ts,
                "gap_minutes": delta.total_seconds() / 60,
                "spans_weekend": spans_weekend,
            }
        )
    return pd.DataFrame(rows)


def check_outliers(df: pd.DataFrame, zscore_threshold: float = 6.0, window: int = 200) -> pd.DataFrame:
    """Flag bars whose return or range is an extreme outlier vs a rolling
    distribution. Useful for spotting bad ticks / fat-finger prints.

    Returns a DataFrame of flagged rows with extra columns: ret, ret_z, range, range_z
    """
    ret = df["close"].pct_change()
    rng = (df["high"] - df["low"]) / df["close"]

    ret_mean = ret.rolling(window, min_periods=30).mean()
    ret_std = ret.rolling(window, min_periods=30).std()
    rng_mean = rng.rolling(window, min_periods=30).mean()
    rng_std = rng.rolling(window, min_periods=30).std()

    ret_z = (ret - ret_mean) / ret_std
    rng_z = (rng - rng_mean) / rng_std

    flagged = (ret_z.abs() > zscore_threshold) | (rng_z.abs() > zscore_threshold)

    out = df.loc[flagged].copy()
    out["ret"] = ret.loc[flagged]
    out["ret_z"] = ret_z.loc[flagged]
    out["range"] = rng.loc[flagged]
    out["range_z"] = rng_z.loc[flagged]
    return out


def check_flat_bars(df: pd.DataFrame, min_run: int = 10) -> pd.DataFrame:
    """Flag runs of consecutive bars with identical OHLC (stale/frozen feed)."""
    flat = (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
    runs = []
    run_start = None
    run_len = 0
    prev_flat = False
    for ts, is_flat in flat.items():
        if is_flat:
            if not prev_flat:
                run_start = ts
                run_len = 1
            else:
                run_len += 1
        else:
            if prev_flat and run_len >= min_run:
                runs.append({"run_start": run_start, "run_end": prev_ts, "length": run_len})
            run_len = 0
        prev_flat = is_flat
        prev_ts = ts
    if prev_flat and run_len >= min_run:
        runs.append({"run_start": run_start, "run_end": prev_ts, "length": run_len})
    return pd.DataFrame(runs)


def summarize(df: pd.DataFrame, expected_minutes: int = 15) -> dict:
    """Run all checks and return a summary dict suitable for printing/saving."""
    dupes = check_duplicates(df)
    bad_ohlc = check_ohlc_consistency(df)
    gaps = check_gaps(df, expected_minutes)
    outliers = check_outliers(df)
    flat_runs = check_flat_bars(df)

    unexpected_gaps = gaps.loc[~gaps["spans_weekend"]] if not gaps.empty else gaps

    return {
        "n_rows": len(df),
        "date_range": (df.index.min(), df.index.max()),
        "n_duplicates": len(dupes),
        "n_bad_ohlc": len(bad_ohlc),
        "n_gaps_total": len(gaps),
        "n_gaps_unexpected": len(unexpected_gaps),
        "n_outliers": len(outliers),
        "n_flat_runs": len(flat_runs),
        "duplicates": dupes,
        "bad_ohlc": bad_ohlc,
        "gaps": gaps,
        "unexpected_gaps": unexpected_gaps,
        "outliers": outliers,
        "flat_runs": flat_runs,
    }
