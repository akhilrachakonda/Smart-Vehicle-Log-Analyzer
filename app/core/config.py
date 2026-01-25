from pathlib import Path

# --- Project Root ---
# Defines the absolute path to the project's root directory.
# This is used to build other paths, ensuring they are always correct,
# regardless of where the application is run from.
PROJECT_ROOT = Path(__file__).parent.parent.parent

# --- ML Model Configuration ---
# This section centralizes the configuration for the machine learning artifacts.
# It's like a manifest that tells the application what files to use.
ML_DIR = PROJECT_ROOT / "app" / "ml"
MODEL_NAME = "anomaly_model.joblib"
SCALER_NAME = "scaler.joblib"
MODEL_PATH = ML_DIR / MODEL_NAME
SCALER_PATH = ML_DIR / SCALER_NAME

# --- Feature Configuration ---
# Defines the list of signals (features) that the model was trained on.
# This is CRITICAL for ensuring that the data fed to the model at inference time
# has the exact same structure as the training data.
# Automotive Analogy: This is like the ECU's CAN signal database (DBC file),
# specifying which signals to read and in what order.
FEATURES = [
    'engine_temp', 
    'vehicle_speed',
    'battery_voltage',
    'brake_pressure'
]
