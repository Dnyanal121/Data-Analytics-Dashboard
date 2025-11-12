import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def perform_action(filepath, action, column=None, extra=None, sample_threshold=500000, model_dir=None):
    """
    Perform a wide range of feature engineering operations on a dataset.
    Supports imputation, encoding, scaling, outlier removal, feature creation, and correlation filtering.
    """
    # Load data
    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)

    original_shape = df.shape

    # ------------------- 1️⃣ Missing Value Imputation -------------------
    if action == "Missing Value Imputation":
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

        msg = f"✅ Missing values imputed successfully! ({len(numeric_cols)} numeric, {len(cat_cols)} categorical columns fixed.)"

    # ------------------- 2️⃣ Drop Column -------------------
    elif action == "Drop Column" and column:
        if column in df.columns:
            df.drop(columns=[column], inplace=True)
            msg = f"✅ Column '{column}' dropped successfully!"
        else:
            return {"message": f"❌ Column '{column}' not found."}

    # ------------------- 3️⃣ Label Encoding -------------------
    elif action == "Label Encoding" and column:
        if column in df.columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            msg = f"✅ Label Encoding applied to '{column}'."

    # ------------------- 4️⃣ One-Hot Encoding -------------------
    elif action == "One-Hot Encoding" and column:
        if column in df.columns:
            df = pd.get_dummies(df, columns=[column])
            msg = f"✅ One-Hot Encoding applied to '{column}'."

    # ------------------- 5️⃣ Feature Scaling -------------------
    elif action == "Feature Scaling" and column:
        if column in df.columns:
            scaler = StandardScaler()
            df[column] = scaler.fit_transform(df[[column]])
            msg = f"✅ Feature Scaling applied to '{column}'."

    # ------------------- 6️⃣ Outlier Removal -------------------
    elif action == "Outlier Removal" and column:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            before = len(df)
            df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
            after = len(df)
            msg = f"✅ Outliers removed from '{column}' — {before - after} rows dropped."

    # ------------------- 7️⃣ Log Transformation -------------------
    elif action == "Log Transformation" and column:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].apply(lambda x: np.log1p(x) if x >= 0 else x)
            msg = f"✅ Log transformation applied to '{column}'."

    # ------------------- 8️⃣ Create New Feature -------------------
    elif action == "Create New Feature" and extra:
        try:
            # Example: new_col = col1 * col2
            parts = extra.split("=")
            new_col = parts[0].strip()
            expr = parts[1].strip()
            df[new_col] = df.eval(expr)
            msg = f"✅ New feature '{new_col}' created using expression: {expr}"
        except Exception as e:
            return {"message": f"❌ Failed to create new feature: {e}"}

    # ------------------- 9️⃣ Correlation-based Feature Selection -------------------
    elif action == "Correlation Filtering":
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_cols = [column for column in upper.columns if any(upper[column] > 0.9)]
        df.drop(columns=drop_cols, inplace=True)
        msg = f"✅ Correlation filtering done. Dropped {len(drop_cols)} highly correlated features: {drop_cols}"

    # ------------------- Default Case -------------------
    else:
        msg = f"⚠️ Action '{action}' not implemented or missing parameters."

    # Save back to file
    if filepath.endswith(".parquet"):
        df.to_parquet(filepath, index=False)
    else:
        df.to_csv(filepath, index=False)

    # Return summary
    sample_html = df.head(10).to_html(classes='table table-sm table-bordered', index=False)
    return {
        "message": msg,
        "shape_change": f"Shape changed from {original_shape} → {df.shape}",
        "sample_head": sample_html
    }
