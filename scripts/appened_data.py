# scripts/append_batch.py
import pandas as pd
from pathlib import Path
import sys, os, subprocess

RAW = Path("data/raw/dataset.csv")
BATCH = Path("data/new_batch/new_batch.csv")

if not BATCH.exists():
    print("No new batch found:", BATCH)
    sys.exit(2)

df = pd.read_csv(RAW)
batch = pd.read_csv(BATCH)

# minimal sanity checks
assert set(batch.columns) == set(df.columns), "Column mismatch"

# append but keep original index
df2 = pd.concat([df, batch], axis=0).reset_index(drop=True)
df2.to_csv(RAW, index=False)
print(f"Appended {len(batch)} rows. New dataset size: {len(df2)}")

# Now update DVC pointer locally (assumes dvc is installed & configured)
# This will create data/raw/dataset.csv.dvc and update cache, but should be committed by the workflow
subprocess.run(["dvc", "add", str(RAW)], check=True)
print("Ran dvc add for", RAW)
