import os
import dask.dataframe as dd
import pandas as pd
import numpy as np

def load_bigdata(filepath):
    """Load dataset with Dask for scalability."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in [".csv", ".txt"]:
            df = dd.read_csv(filepath, assume_missing=True, low_memory=False)
        elif ext in [".parquet"]:
            df = dd.read_parquet(filepath)
        else:
            df = dd.read_csv(filepath, assume_missing=True)
        return df
    except Exception as e:
        print("⚠️ Dask failed, falling back to pandas:", e)
        return pd.read_csv(filepath)

def get_columns(df):
    """Return column list."""
    try:
        return list(df.columns)
    except:
        return []

def get_sample(df, sample_threshold=500000):
    """Return sample Pandas DataFrame for fast operations."""
    try:
        if hasattr(df, "compute"):
            n = len(df)
            if n > sample_threshold:
                frac = sample_threshold / n
                return df.sample(frac=frac).compute()
            return df.compute()
        return df
    except:
        return df


def clean_null_values(df):
    """
    Converts 'nan', 'NaN', 'NULL', 'null', and empty strings to proper NaN.
    Works for both Dask and Pandas DataFrames.
    """
    null_like = ["nan", "NaN", "NULL", "null", "", " "]

    if isinstance(df, dd.DataFrame):
        for col in df.columns:
            df[col] = df[col].map_partitions(
                lambda s: s.replace(null_like, np.nan)
            )
    else:
        df = df.replace(null_like, np.nan)
    
    return df