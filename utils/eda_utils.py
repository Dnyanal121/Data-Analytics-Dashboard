import pandas as pd
import numpy as np
import os
from .data_utils import get_sample


# -------------------- 🔹 Helper: Clean and Standardize Missing Values -------------------- #
def clean_missing_values(df):
    """
    Converts various string-based null indicators to real NaN values.
    """
    null_like_values = [
        "na", "n/a", "nan", "none", "null", "?", "-", "--", "missing", "blank", ""
    ]
    df = df.replace(null_like_values, np.nan)
    return df


# -------------------- 🔹 Core EDA Functions -------------------- #

def basic_eda(df):
    """
    Returns basic descriptive stats, missing values %, and dtypes for quick overview.
    """
    df = clean_missing_values(df)
    df_sample = get_sample(df)

    desc_html = df_sample.describe(include="all").to_html(classes="table table-sm table-striped")
    missing = df_sample.isna().mean().round(3) * 100
    missing_dict = missing.to_dict()
    dtypes = df_sample.dtypes.astype(str).to_dict()

    return desc_html, missing_dict, dtypes


def head_html(df, n=10):
    """
    Returns the top N rows as HTML.
    """
    df = clean_missing_values(df)
    df_sample = get_sample(df)
    return df_sample.head(n).to_html(classes="table table-bordered table-sm", index=False)


def dashboard_summary(df):
    """
    Generates high-level dashboard metrics about the dataset.
    """
    df = clean_missing_values(df)
    df_sample = get_sample(df)
    stats = {
        "rows": len(df_sample),
        "columns": len(df_sample.columns),
        "missing_percentage": float(df_sample.isna().mean().mean() * 100),
        "numeric_cols": len(df_sample.select_dtypes(include=np.number).columns),
        "categorical_cols": len(df_sample.select_dtypes(exclude=np.number).columns),
        "duplicate_rows": int(df_sample.duplicated().sum()),
        "memory_usage_MB": round(df_sample.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }
    return stats


# -------------------- 🔹 Extended Deep EDA Section -------------------- #

def numeric_analysis(df):
    df = clean_missing_values(df)
    df_sample = get_sample(df)
    num_df = df_sample.select_dtypes(include=np.number)

    if num_df.empty:
        return pd.DataFrame(columns=["column", "mean", "std", "min", "max", "skew", "kurtosis"]).to_html(classes="table")

    summary = pd.DataFrame({
        "mean": num_df.mean(),
        "std": num_df.std(),
        "min": num_df.min(),
        "max": num_df.max(),
        "skew": num_df.skew(),
        "kurtosis": num_df.kurt()
    }).round(3)

    return summary.to_html(classes="table table-sm table-striped table-hover")


def categorical_analysis(df):
    df = clean_missing_values(df)
    df_sample = get_sample(df)
    cat_df = df_sample.select_dtypes(exclude=np.number)

    if cat_df.empty:
        return "<p>No categorical columns found.</p>"

    freq_tables = ""
    for col in cat_df.columns[:10]:
        value_counts = cat_df[col].value_counts(dropna=False).head(10)
        temp_df = pd.DataFrame({
            "Category": value_counts.index.astype(str),
            "Count": value_counts.values,
            "Percentage": np.round((value_counts.values / len(cat_df)) * 100, 2)
        })
        freq_tables += f"<h6>{col}</h6>" + temp_df.to_html(classes="table table-bordered table-sm", index=False) + "<br>"

    return freq_tables


def correlation_analysis(df):
    df = clean_missing_values(df)
    df_sample = get_sample(df)
    num_df = df_sample.select_dtypes(include=np.number)

    if num_df.empty or num_df.shape[1] < 2:
        return "<p>Not enough numeric columns for correlation analysis.</p>"

    corr = num_df.corr().round(3)
    corr_html = corr.to_html(classes="table table-sm table-striped table-hover")
    top_corr = (
        corr.unstack()
        .reset_index()
        .rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: "Correlation"})
    )
    top_corr = top_corr[top_corr["Feature_1"] != top_corr["Feature_2"]]
    top_corr = top_corr.reindex(top_corr["Correlation"].abs().sort_values(ascending=False).index).head(10)
    top_corr_html = top_corr.to_html(classes="table table-sm table-bordered", index=False)

    return f"<h5>Correlation Matrix</h5>{corr_html}<br><h5>Top 10 Correlated Feature Pairs</h5>{top_corr_html}"


def imbalance_check(df, target_col=None):
    df = clean_missing_values(df)
    if target_col is None or target_col not in df.columns:
        return "<p>No target column specified or found for imbalance check.</p>"

    counts = df[target_col].value_counts()
    imbalance_df = pd.DataFrame({
        "Class": counts.index.astype(str),
        "Count": counts.values,
        "Percentage": np.round((counts.values / len(df)) * 100, 2)
    })

    return imbalance_df.to_html(classes="table table-bordered table-sm", index=False)


def deep_eda_summary(df, target_col=None):
    summary = {
        "basic_stats": basic_eda(df)[0],
        "numeric_summary": numeric_analysis(df),
        "categorical_summary": categorical_analysis(df),
        "correlations": correlation_analysis(df),
        "imbalance": imbalance_check(df, target_col)
    }
    return summary


# -------------------- 🔹 Parquet Conversion + Save Final Dataset -------------------- #

def save_cleaned_dataset(df, filename):
    """
    Saves cleaned and processed dataset to Parquet format after preprocessing.
    """
    df = clean_missing_values(df)
    df = df.drop_duplicates()

    # create folder if not exists
    output_folder = "processed_datasets"
    os.makedirs(output_folder, exist_ok=True)

    parquet_path = os.path.join(output_folder, f"{filename}.parquet")
    df.to_parquet(parquet_path, index=False)

    return parquet_path
