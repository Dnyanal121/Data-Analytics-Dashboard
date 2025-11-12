import os
import time
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, send_file, jsonify
)
from werkzeug.utils import secure_filename

# Utils imports
from utils import data_utils, eda_utils, feature_engineering, visualization_utils, modelling_utils

# --- Config ---
app = Flask(__name__)
app.secret_key = "replace_with_a_secure_random_string"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMAGE_DIR = os.path.join(BASE_DIR, "static", "images")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_datasets")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Maximum rows to load fully; beyond this, Dask sampling kicks in
DEFAULT_SAMPLE_THRESHOLD = 500000

# -----------------------------------------------------------
# -------------------- ROUTES SECTION ------------------------
# -----------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Landing page: dataset upload (CSV / Excel / Parquet / TSV).
    Converts everything internally to Parquet.
    """
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file:
            flash("Please upload a file", "danger")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)

        # ---- Convert any supported file to Parquet ----
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(filepath)
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                df = pd.read_excel(filepath)
            elif filename.endswith(".tsv"):
                df = pd.read_csv(filepath, sep="\t")
            elif filename.endswith(".parquet"):
                df = pd.read_parquet(filepath)
            else:
                flash("❌ Unsupported file format!", "danger")
                return redirect(url_for("index"))

            # Convert to Parquet
            parquet_name = os.path.splitext(filename)[0] + ".parquet"
            parquet_path = os.path.join(UPLOAD_DIR, parquet_name)
            df.to_parquet(parquet_path, index=False)

            # Save Parquet path to session
            session["filepath"] = parquet_path
            flash(f"✅ File converted and saved as {parquet_name}", "success")
            return redirect(url_for("operation"))

        except Exception as e:
            flash(f"❌ Failed to process file: {str(e)}", "danger")
            return redirect(url_for("index"))

    return render_template("index.html")


@app.route("/operation")
def operation():
    """
    Operation dashboard after dataset upload.
    """
    if "filepath" not in session:
        flash("Upload a dataset first", "warning")
        return redirect(url_for("index"))
    return render_template("operation.html")

# -----------------------------------------------------------
# -------------------- EDA ROUTE -----------------------------
# -----------------------------------------------------------

@app.route("/eda", methods=["GET", "POST"])
def eda():
    """
    Perform basic and deep EDA.
    """
    if "filepath" not in session:
        flash("Upload dataset first", "warning")
        return redirect(url_for("index"))

    filepath = session["filepath"]
    df = data_utils.load_bigdata(filepath)  # uses Dask if large

    if request.method == "POST":
        target_col = request.form.get("target_col") or None
        eda_results = eda_utils.deep_eda_summary(df, target_col=target_col)
        return render_template("eda.html", eda=eda_results)

    # GET request — just load file and basic info
    desc_html, missing_dict, dtypes = eda_utils.basic_eda(df)
    head_html = eda_utils.head_html(df, n=10)
    stats = eda_utils.dashboard_summary(df)
    return render_template(
        "eda.html",
        summary=desc_html,
        missing=missing_dict,
        dtypes=dtypes,
        head=head_html,
        stats=stats
    )


# -----------------------------------------------------------
# ---------------- FEATURE ENGINEERING -----------------------
# -----------------------------------------------------------

@app.route("/feature-engineering", methods=["GET", "POST"])
def feature_engineering_page():
    """
    Handles feature transformations, encoding, scaling, etc.
    """
    if "filepath" not in session:
        flash("Upload dataset first", "warning")
        return redirect(url_for("index"))

    filepath = session["filepath"]
    df = data_utils.load_bigdata(filepath)
    cols = data_utils.get_columns(df)
    result = None

    if request.method == "POST":
        action = request.form.get("action")
        col = request.form.get("column")
        extra = request.form.get("extra")

        result = feature_engineering.perform_action(
            filepath, action, column=col, extra=extra,
            sample_threshold=DEFAULT_SAMPLE_THRESHOLD,
            model_dir=MODEL_DIR
        )

        flash(f"✅ Feature Engineering '{action}' completed successfully!", "success")

    return render_template("feature_engineering.html", cols=cols, result=result)


# -----------------------------------------------------------
# ---------------- VISUALIZATION -----------------------------
# -----------------------------------------------------------

@app.route("/visualization", methods=["GET", "POST"])
def visualization_page():
    """
    Handles chart generation and data visual exploration.
    """
    if "filepath" not in session:
        flash("Upload dataset first", "warning")
        return redirect(url_for("index"))

    filepath = session["filepath"]
    df = data_utils.load_bigdata(filepath)
    cols = data_utils.get_columns(df)
    plot_html = None

    if request.method == "POST":
        x = request.form.get("x")
        y = request.form.get("y")
        chart_type = request.form.get("chart_type")

        plot_html = visualization_utils.generate_plot(
            df, x, y, chart_type, sample_threshold=DEFAULT_SAMPLE_THRESHOLD
        )

    return render_template("visualization.html", cols=cols, plot_html=plot_html)


# -----------------------------------------------------------
# ---------------- MODELLING SECTION ------------------------
# -----------------------------------------------------------

@app.route("/modelling", methods=["GET", "POST"])
def modelling_page():
    """
    Model training and evaluation interface.
    """
    if "filepath" not in session:
        flash("Upload dataset first", "warning")
        return redirect(url_for("index"))

    filepath = session["filepath"]
    df = data_utils.load_bigdata(filepath)
    cols = data_utils.get_columns(df)
    metrics, model_info = None, None

    if request.method == "POST":
        algo = request.form.get("algo")
        target = request.form.get("target")
        sample_frac = float(request.form.get("sample_frac") or 0.05)

        model_info, metrics = modelling_utils.train_model(
            filepath, target, algo,
            sample_frac=sample_frac,
            sample_threshold=DEFAULT_SAMPLE_THRESHOLD,
            model_dir=MODEL_DIR
        )

        if model_info and "model_path" in model_info:
            session["latest_model_path"] = model_info["model_path"]
            session["latest_feature_columns"] = model_info["feature_columns"]

        flash("✅ Model training completed successfully!", "success")

    return render_template("modelling.html", cols=cols, metrics=metrics, model_info=model_info)


# -----------------------------------------------------------
# ---------------- PREDICTION SECTION -----------------------
# -----------------------------------------------------------

@app.route("/prediction", methods=["GET", "POST"])
def prediction_page():
    """
    Predict using the last trained model.
    """
    if "latest_model_path" not in session:
        flash("Train a model first under Modelling", "warning")
        return redirect(url_for("modelling_page"))

    model_path = session["latest_model_path"]
    feature_cols = session.get("latest_feature_columns", [])
    result_table = None

    if request.method == "POST":
        file = request.files.get("prediction_file")

        # Batch prediction from CSV/Parquet
        if file and file.filename:
            upload_name = secure_filename(file.filename)
            pred_path = os.path.join(UPLOAD_DIR, f"pred_input_{int(time.time())}_{upload_name}")
            file.save(pred_path)

            out_df = modelling_utils.predict_batch(pred_path, model_path, feature_cols)
            out_csv = os.path.join(UPLOAD_DIR, f"pred_out_{int(time.time())}.csv")
            out_df.to_csv(out_csv, index=False)

            result_table = out_df.head(200).to_html(classes="table table-striped table-sm", index=False)
            session["last_prediction_file"] = out_csv

        # Single-row manual prediction
        else:
            input_data = {}
            for c in feature_cols:
                val = request.form.get(f"feature__{c}")
                if val == "" or val is None:
                    input_data[c] = None
                else:
                    try:
                        input_data[c] = float(val)
                    except ValueError:
                        input_data[c] = val

            pred = modelling_utils.predict_single(input_data, model_path, feature_cols)
            result_table = f"<table class='table table-sm'><tr><th>Prediction</th></tr><tr><td>{pred}</td></tr></table>"

    return render_template("prediction.html", cols=feature_cols, result_table=result_table)


@app.route("/download_predictions")
def download_predictions():
    """
    Download prediction output CSV.
    """
    f = session.get("last_prediction_file")
    if not f or not os.path.exists(f):
        flash("No prediction output file available", "warning")
        return redirect(url_for("prediction_page"))
    return send_file(f, as_attachment=True)


# -----------------------------------------------------------
# ---------------- DASHBOARD METRICS ------------------------
# -----------------------------------------------------------

@app.route("/dashboard")
def dashboard_page():
    """
    Lightweight dashboard of key dataset metrics.
    """
    if "filepath" not in session:
        flash("Upload dataset first", "warning")
        return redirect(url_for("index"))

    filepath = session["filepath"]
    df = data_utils.load_bigdata(filepath)
    stats = eda_utils.dashboard_summary(df)
    return render_template("dashboard.html", stats=stats)


# -----------------------------------------------------------
# ---------------- HEALTH CHECK -----------------------------
# -----------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# -----------------------------------------------------------
# ---------------- RUN SERVER -------------------------------
# -----------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
