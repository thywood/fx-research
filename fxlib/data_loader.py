"""Load and clean OHLCV CSV exports from MT5's History Center.

MT5 exports vary in format:
  - Separator: tab or comma
  - Header: present (e.g. <DATE> <TIME> <OPEN> ...) or absent
  - Date/Time: combined "DateTime" column, or separate Date + Time columns
  - Volume: "tick volume" only, or tick + real volume, sometimes + spread

This loader auto-detects the common variants and returns a normalized
DataFrame indexed by UTC-naive datetime with columns:
  open, high, low, close, volume
"""

from __future__ import annotations

import csv
import re

import pandas as pd


_CANDIDATE_COL_MAP = {
    "date": "date",
    "<date>": "date",
    "time": "time",
    "<time>": "time",
    "open": "open",
    "<open>": "open",
    "high": "high",
    "<high>": "high",
    "low": "low",
    "<low>": "low",
    "close": "close",
    "<close>": "close",
    "tickvol": "volume",
    "<tickvol>": "volume",
    "vol": "volume",
    "<vol>": "volume",
    "volume": "volume",
    "spread": "spread",
    "<spread>": "spread",
}


def _detect_separator(sample_line: str) -> str:
    if "\t" in sample_line:
        return "\t"
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample_line)
        return dialect.delimiter
    except csv.Error:
        return ","


def _looks_like_header(first_cell: str) -> bool:
    first_cell = first_cell.strip().lower()
    return first_cell in _CANDIDATE_COL_MAP or not re.match(r"^\d", first_cell)


def load_mt5_csv(path: str) -> pd.DataFrame:
    """Load an MT5 History Center OHLCV export into a clean DataFrame.

    Returns a DataFrame indexed by datetime (naive, broker server time as
    exported by MT5) with columns: open, high, low, close, volume.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        first_line = f.readline()

    sep = _detect_separator(first_line)
    first_cell = first_line.split(sep)[0]
    has_header = _looks_like_header(first_cell)

    raw = pd.read_csv(
        path,
        sep=sep,
        header=0 if has_header else None,
        encoding="utf-8-sig",
        dtype=str,
    )

    if has_header:
        normalized = []
        for col in raw.columns:
            key = str(col).strip().lower()
            normalized.append(_CANDIDATE_COL_MAP.get(key, key))
        raw.columns = normalized
    else:
        # Positional fallback: DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOLUME, ...
        n = raw.shape[1]
        base = ["date", "time", "open", "high", "low", "close", "volume"]
        raw.columns = base[:n] + [f"extra{i}" for i in range(n - len(base))] if n > len(base) else base[:n]

    if "date" in raw.columns and "time" in raw.columns:
        dt = pd.to_datetime(
            raw["date"].str.replace(".", "-", regex=False) + " " + raw["time"],
            errors="raise",
        )
    elif "date" in raw.columns:
        # Single combined date/time column labeled "date"
        dt = pd.to_datetime(raw["date"].str.replace(".", "-", regex=False), errors="raise")
    else:
        raise ValueError(f"Could not find date/time columns in {path}; found columns: {list(raw.columns)}")

    # Use .to_numpy() throughout: dt and the raw columns share the same
    # default RangeIndex, but that index has no relation to the new
    # DatetimeIndex we're assigning -- building the DataFrame from arrays
    # avoids a spurious index-alignment reindex (which would NaN everything).
    out = pd.DataFrame(
        {
            "open": raw["open"].astype(float).to_numpy(),
            "high": raw["high"].astype(float).to_numpy(),
            "low": raw["low"].astype(float).to_numpy(),
            "close": raw["close"].astype(float).to_numpy(),
            "volume": raw["volume"].astype(float).to_numpy() if "volume" in raw.columns else float("nan"),
        },
        index=pd.DatetimeIndex(dt.to_numpy()),
    )
    out.index.name = "datetime"
    out = out.sort_index()
    return out
