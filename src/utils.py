# src/utils.py
import os, re, tempfile, subprocess
import numpy as np
import pandas as pd

def download_from_gdrive(link: str, out_csv: str) -> str:
    """
    Supports a shared GDrive link; uses gdown which handles various formats.
    """
    import gdown
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    gdown.download(url=link, output=out_csv, quiet=False, fuzzy=True)
    return out_csv

def apply_poisoning(df: pd.DataFrame, target: str, problem_type: str, spec: str) -> pd.DataFrame:
    """
    spec: 'flip:0.1' (flip 10% labels) for classification
          'noise:0.1' (add Gaussian noise to 10% rows for all numeric features) for regression/classification
    """
    df = df.copy()
    mode, frac = spec.split(":")
    frac = float(frac)
    n = int(len(df) * frac)
    idx = np.random.choice(df.index, size=n, replace=False)

    if mode == "flip" and problem_type == "classification":
    # flip labels among observed unique labels
        unique = df[target].unique()
        if len(unique) == 2:
        # Handle both numeric {0,1} and string {"yes","no"} targets
            if set(unique) <= {0, 1}:
                df.loc[idx, target] = 1 - df.loc[idx, target]
            elif set(unique) <= {"yes", "no"}:
                mapping = {"yes": "no", "no": "yes"}
                df.loc[idx, target] = df.loc[idx, target].map(mapping)
            else:
                raise ValueError(f"Unsupported binary labels: {unique}")
        else:
        # multi-class: rotate labels
            mapping = {c: i for i, c in enumerate(unique)}
            inv = {i: c for c, i in mapping.items()}
            df.loc[idx, target] = [inv[(mapping[v] + 1) % len(unique)] for v in df.loc[idx, target]]

    return df
