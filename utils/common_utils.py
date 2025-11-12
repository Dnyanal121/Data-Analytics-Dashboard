import dask.dataframe as dd

def get_sample(df, n=1000):
    """
    Returns a small sample of a large Dask or Pandas DataFrame.
    Useful for fast EDA.
    """
    try:
        if isinstance(df, dd.DataFrame):
            return df.sample(frac=min(1.0, n / len(df))).compute()
        else:
            return df.sample(n=min(n, len(df)))
    except Exception:
        return df.head(n)
