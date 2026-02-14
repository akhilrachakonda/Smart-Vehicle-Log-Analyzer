import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

# Import the centralized configuration
from app.core.config import MODEL_PATH, SCALER_PATH, FEATURES, MODEL_CONFIGS, ML_DIR

# --- Model Loading ---
# This section handles loading the pre-trained model and scaler from the disk.
# This is done once when the module is first imported, making it efficient.
# The application will not re-load the model for every request.
# Automotive Analogy: This is the ECU's Power-On Self-Test (POST), where it loads
# its calibration data from non-volatile memory into active RAM.

model = None
scaler = None

try:
    print(f"Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    model = None # Ensure model is None if loading fails

try:
    print(f"Loading scaler from: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully.")
except FileNotFoundError:
    print(f"Error: Scaler file not found at {SCALER_PATH}")
    scaler = None # Ensure scaler is None if loading fails

def get_model_artifacts():
    """
    Returns the loaded model and scaler.
    Raises a RuntimeError if artifacts are not loaded.
    """
    if model is None or scaler is None:
        raise RuntimeError(
            "Model or scaler artifacts are not loaded. "
            "Please ensure the training script has been run and the artifacts are in the correct path."
        )
    return model, scaler

# --- Prediction Function ---
def predict_anomalies(df: pd.DataFrame) -> np.ndarray:
    """
    Performs anomaly detection on the given dataframe using the default model.

    Args:
        df (pd.DataFrame): The input data, which must contain the columns defined in FEATURES.

    Returns:
        np.ndarray: An array of predictions. '-1' indicates an anomaly, '1' indicates normal.
    """
    model_instance, scaler_instance = get_model_artifacts()

    # 1. Ensure the dataframe has the correct features
    # This is a critical safety check.
    if not all(feature in df.columns for feature in FEATURES):
        missing = [f for f in FEATURES if f not in df.columns]
        raise ValueError(f"Input data is missing required features: {missing}")
    
    # 2. Select and reorder features to match the training order
    X = df[FEATURES]

    # 3. Scale the data using the EXACT SAME scaler from training
    # Automotive Analogy: This is like converting a raw sensor voltage into a physical
    # value (e.g., degrees Celsius) using the same calibration curve defined during development.
    X_scaled = scaler_instance.transform(X)

    # 4. Make predictions
    # The model returns -1 for anomalies and 1 for normal data points.
    predictions = model_instance.predict(X_scaled)

    return predictions

def predict_anomalies_for_model(df: pd.DataFrame, model_name: str) -> np.ndarray:
    """
    Performs anomaly detection using a dynamically specified model.

    Args:
        df (pd.DataFrame): The input data.
        model_name (str): The name of the model config to use (e.g., 'telematics').

    Returns:
        np.ndarray: An array of predictions. '-1' indicates an anomaly, '1' indicates normal.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model configuration '{model_name}' not found.")

    config = MODEL_CONFIGS[model_name]
    model_path = ML_DIR / config["model_name"]
    scaler_path = ML_DIR / config["scaler_name"]
    features = config["features"]

    try:
        model_instance = joblib.load(model_path)
        scaler_instance = joblib.load(scaler_path)
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not load model/scaler for '{model_name}'. Have you trained it? Details: {e}")

    if not all(feature in df.columns for feature in features):
        missing = [f for f in features if f not in df.columns]
        raise ValueError(f"Input data for model '{model_name}' is missing features: {missing}")

    X = df[features]
    X_scaled = scaler_instance.transform(X)
    predictions = model_instance.predict(X_scaled)

    return predictions
