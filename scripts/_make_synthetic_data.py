"""DEV ONLY: generate a small synthetic EUR/USD M15 CSV (with a few
deliberate data-quality issues injected) so the pipeline can be smoke-tested
before real MT5 export data is available. Not used for any real research
conclusions.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

start = pd.Timestamp("2024-01-01 00:00:00")
end = pd.Timestamp("2024-04-01 00:00:00")

all_times = pd.date_range(start, end, freq="15min", inclusive="left")
times = all_times[all_times.weekday < 5]
n = len(times)

returns = np.random.normal(0, 0.0003, n)
close = 1.0800 * np.exp(np.cumsum(returns))

open_ = np.empty(n)
open_[0] = close[0]
open_[1:] = close[:-1]

intrabar_range = np.abs(np.random.normal(0, 0.00015, n))
high = np.maximum(open_, close) + intrabar_range
low = np.minimum(open_, close) - intrabar_range
volume = np.random.randint(50, 500, n)

df = pd.DataFrame(
    {
        "Date": times.strftime("%Y.%m.%d"),
        "Time": times.strftime("%H:%M:%S"),
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }
)

# Inject quality issues for testing fxlib.quality:
df.loc[500, "High"] *= 1.05   # outlier spike
df.loc[500, "Close"] *= 1.03
dup_row = df.iloc[[100]]
df = pd.concat([df.iloc[:101], dup_row, df.iloc[101:]], ignore_index=True)  # duplicate timestamp
df = df.drop(index=[1000, 1001, 1002]).reset_index(drop=True)  # mid-week gap

df.to_csv("data/_synthetic_EURUSD_M15.csv", index=False)
print(f"Wrote {len(df)} rows to data/_synthetic_EURUSD_M15.csv")
