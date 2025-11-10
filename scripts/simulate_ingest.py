# scripts/simulate_ingest.py
import pandas as pd
import numpy as np
import os
from pathlib import Path

Path("data/raw").mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/raw/dataset.csv")  # your current dataset
# create a "new batch" by sampling plus small perturbation to simulate drift
batch = df.sample(30, replace=True).reset_index(drop=True)

# Introduce drift: increase 'age' by +2 for half rows (example)
if 'age' in batch.columns:
    idx = np.random.choice(batch.index, size=len(batch)//2, replace=False)
    batch.loc[idx, 'age'] = batch.loc[idx, 'age'] + 5

batch_path = "data/new_batch/new_batch.csv"
Path("data/new_batch").mkdir(parents=True, exist_ok=True)
batch.to_csv(batch_path, index=False)
print("Wrote new batch to", batch_path)

# Optionally append to main dataset for simulating ingestion:
append = True
if append:
    df2 = pd.concat([df, batch], axis=0).reset_index(drop=True)
    df2.to_csv("data/raw/dataset.csv", index=False)
    print("Appended batch to data/raw/dataset.csv")
