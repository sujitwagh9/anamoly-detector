from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict
import logging

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ANOMALIES_FILE = "anomalies.csv"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 📌 Anomaly Detection Methods
def z_score_method(df: pd.DataFrame, target_col: str, threshold: float = 3) -> pd.Series:
    """Detect anomalies using Z-score method."""
    return abs(zscore(df[target_col])) > threshold

def moving_average_method(df: pd.DataFrame, target_col: str, window: int = 5, threshold: float = 3) -> pd.Series:
    """Detect anomalies using Moving Average with Standard Deviation."""
    rolling_mean = df[target_col].rolling(window=window).mean()
    rolling_std = df[target_col].rolling(window=window).std()
    return (df[target_col] > rolling_mean + threshold * rolling_std) | \
           (df[target_col] < rolling_mean - threshold * rolling_std)

def iqr_method(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Detect anomalies using Interquartile Range (IQR)."""
    Q1, Q3 = df[target_col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return (df[target_col] < lower_bound) | (df[target_col] > upper_bound)

def isolation_forest_method(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Detect anomalies using Isolation Forest."""
    model = IsolationForest(contamination=0.05, random_state=42)
    return pd.Series(model.fit_predict(df[[target_col]]) == -1, index=df.index)

def detect_anomalies(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, str, Dict[str, int]]:
    """Detect anomalies and select the best method based on 5% anomaly target."""
    methods = {
        "Z-Score": z_score_method(df, target_col),
        "Moving Average": moving_average_method(df, target_col),
        "IQR": iqr_method(df, target_col),
        "Isolation Forest": isolation_forest_method(df, target_col),
    }

    anomaly_counts = {method: int(anomalies.sum()) for method, anomalies in methods.items()}
    target_count = len(df) * 0.05
    best_method = min(anomaly_counts, key=lambda k: abs(anomaly_counts[k] - target_count))

    df["anomaly"] = methods[best_method]
    return df, best_method, anomaly_counts

# 📌 Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/api/anomalies", methods=["GET"])
def get_anomalies():
    if not os.path.exists(ANOMALIES_FILE):
        return jsonify({"data": [], "total_anomalies": 0, "total_data": 0, "best_method": "Unknown"})
    
    df = pd.read_csv(ANOMALIES_FILE)
    return jsonify({
        "data": df.to_dict(orient="records"),
        "total_anomalies": int(df["anomaly"].sum()),
        "total_data": len(df),
        "best_method": df.attrs.get("best_method", "Unknown")
    })

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files.get("file")
    if not file:
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return "Invalid CSV file", 400

    if df.empty:
        return "Uploaded file is empty", 400

    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numerical_cols:
        return "No numerical columns found", 400

    target_col = numerical_cols[0]
    df = df.dropna(subset=[target_col])

    # Add a date-like index if not present (fixed AttributeError issue)
    if not any(col.lower() == "date" for col in df.columns):
        df["date"] = pd.date_range(start="2023-01-01", periods=len(df), freq="D").strftime("%Y-%m-%d")

    df, best_method, method_results = detect_anomalies(df, target_col)
    df.attrs["best_method"] = best_method
    df.to_csv(ANOMALIES_FILE, index=False)

    return render_template(
        "detect.html",
        data=df.to_dict(orient="records"),
        dates=df["date"].tolist(),  # For the line chart
        values=df[target_col].tolist(),  # For the line chart
        anomalies=df[df["anomaly"]].to_dict(orient="records"),  # Pass anomalies separately
        best_method=best_method,
        target_col=target_col
    )

@app.route("/download/anomalies.csv")
def download_anomalies():
    if not os.path.exists(ANOMALIES_FILE):
        return "No anomalies detected yet. Please run anomaly detection first.", 404
    return send_file(ANOMALIES_FILE, as_attachment=True, mimetype="text/csv")


@app.route("/contact", methods=["POST"])
def contact():
    name = request.form.get("name")
    email = request.form.get("email")
    message = request.form.get("message")
    # Process the data (e.g., save to DB, send email)
    return "Message sent!"


if __name__ == "__main__":
    app.run(debug=True)