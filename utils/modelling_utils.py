import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from .data_utils import get_sample, load_bigdata

def train_model(filepath, target, algo, sample_frac=0.05, sample_threshold=500000, model_dir=None):
    df = load_bigdata(filepath)
    df = get_sample(df, sample_threshold)
    if target not in df.columns:
        return None, {"error": f"Target column '{target}' not found"}

    X = df.drop(columns=[target])
    y = df[target]
    X = pd.get_dummies(X, drop_first=True)

    # Pick model type
    if algo.lower() in ["linear_regression", "linreg"]:
        model = LinearRegression()
        model_type = "regression"
    elif algo.lower() in ["logistic_regression", "logreg"]:
        model = LogisticRegression(max_iter=1000)
        model_type = "classification"
    elif algo.lower() == "random_forest_classifier":
        model = RandomForestClassifier()
        model_type = "classification"
    elif algo.lower() == "random_forest_regressor":
        model = RandomForestRegressor()
        model_type = "regression"
    elif algo.lower() == "xgboost_classifier":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model_type = "classification"
    else:
        model = XGBRegressor()
        model_type = "regression"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {}
    if model_type == "classification":
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds, average="weighted"),
            "precision": precision_score(y_test, preds, average="weighted"),
            "recall": recall_score(y_test, preds, average="weighted"),
        }
    else:
        metrics = {"r2": r2_score(y_test, preds)}

    model_path = os.path.join(model_dir, f"{algo}_{target}.pkl")
    joblib.dump(model, model_path)

    model_info = {"model_path": model_path, "feature_columns": list(X.columns), "algo": algo}
    return model_info, metrics


def predict_batch(pred_path, model_path, feature_cols):
    model = joblib.load(model_path)
    df = pd.read_csv(pred_path)
    df = pd.get_dummies(df)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    preds = model.predict(df[feature_cols])
    df["prediction"] = preds
    return df


def predict_single(input_data, model_path, feature_cols):
    model = joblib.load(model_path)
    df = pd.DataFrame([input_data])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    preds = model.predict(df[feature_cols])
    return preds[0]
