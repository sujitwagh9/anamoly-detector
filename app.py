from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
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

def create_sequences(data: np.ndarray, seq_length: int = 10) -> np.ndarray:
    """Create sequences for LSTM input."""
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

def lstm_method(df: pd.DataFrame, target_col: str, seq_length: int = 10, epochs: int = 20) -> pd.Series:
    """Detect anomalies using LSTM Autoencoder."""
    try:
        # Preprocess data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[[target_col]].values)
        
        # Create sequences
        sequences = create_sequences(scaled_data, seq_length)
        if len(sequences) < 1:
            logger.warning("Not enough data for LSTM sequences")
            return pd.Series([False] * len(df), index=df.index)

        # Build LSTM Autoencoder
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(seq_length, 1), return_sequences=False),
            RepeatVector(seq_length),
            LSTM(64, activation='relu', return_sequences=True),
            TimeDistributed(Dense(1))
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Reshape for LSTM [samples, timesteps, features]
        X = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
        
        # Train model
        model.fit(X, X, epochs=epochs, batch_size=32, verbose=0)
        
        # Predict and calculate reconstruction error
        preds = model.predict(X, verbose=0)
        mse = np.mean(np.power(X - preds, 2), axis=(1, 2))
        
        # Determine threshold for anomalies (mean + 2 * std)
        threshold = np.mean(mse) + 2 * np.std(mse)
        anomalies = mse > threshold
        
        # Align anomalies with original dataframe
        result = [False] * len(df)
        for i, is_anomaly in enumerate(anomalies):
            result[i + seq_length - 1] = is_anomaly
        
        return pd.Series(result, index=df.index)
    except Exception as e:
        logger.error(f"LSTM method failed: {e}")
        return pd.Series([False] * len(df), index=df.index)

def detect_anomalies(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, str, Dict[str, int]]:
    """Detect anomalies and select the best method based on 5% anomaly target."""
    methods = {
        "Z-Score": z_score_method(df, target_col),
        "Moving Average": moving_average_method(df, target_col),
        "IQR": iqr_method(df, target_col),
        "Isolation Forest": isolation_forest_method(df, target_col),
        "LSTM": lstm_method(df, target_col)
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

    # Add a date-like index if not present
    if not any(col.lower() == "date" for col in df.columns):
        df["date"] = pd.date_range(start="2023-01-01", periods=len(df), freq="D").strftime("%Y-%m-%d")

    df, best_method, method_results = detect_anomalies(df, target_col)
    df.attrs["best_method"] = best_method
    df.to_csv(ANOMALIES_FILE, index=False)

    # Prepare data for visualization and table
    anomaly_indices = df[df["anomaly"]].index.tolist()
    anomaly_values = df[df["anomaly"]][target_col].tolist()
    anomaly_dates = df[df["anomaly"]]["date"].tolist()
    anomaly_data = df[df["anomaly"]].to_dict(orient="records")
    columns = df.columns.tolist()

    return render_template(
        "detect.html",
        data=df.to_dict(orient="records"),
        dates=df["date"].tolist(),
        values=df[target_col].tolist(),
        anomalies=anomaly_data,
        best_method=best_method,
        target_col=target_col,
        anomaly_indices=anomaly_indices,
        anomaly_values=anomaly_values,
        anomaly_dates=anomaly_dates,
        total_anomalies=len(anomaly_indices),
        total_data=len(df),
        method_results=method_results,
        columns=columns
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