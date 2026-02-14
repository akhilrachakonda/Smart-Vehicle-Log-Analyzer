from pathlib import Path

# --- Project Root ---
# Defines the absolute path to the project's root directory.
# This is used to build other paths, ensuring they are always correct,
# regardless of where the application is run from.
PROJECT_ROOT = Path(__file__).parent.parent.parent

# --- ML Model Configuration ---
# This section centralizes the configuration for multiple machine learning models.
# It allows the application to be flexible and switch between different models.
ML_DIR = PROJECT_ROOT / "app" / "ml"

MODEL_CONFIGS = {
    "synthetic": {
        "model_name": "anomaly_model.joblib",
        "scaler_name": "scaler.joblib",
        "features": [
            'engine_temp', 
            'vehicle_speed',
            'battery_voltage',
            'brake_pressure'
        ]
    },
    "telematics": {
        "model_name": "telematics_anomaly_model.joblib",
        "scaler_name": "telematics_scaler.joblib",
        "features": [
            'air_intake_temperature',
            'calculated_engine_load',
            'control_module_voltage',
            'engine_coolant_temperature',
            'engine_rpm',
            'intake_manifold_absolute_pressure',
            'throttle_position',
            'vehicle_speed'
        ]
    }
}

# Define the default model to be loaded on application startup
DEFAULT_MODEL_NAME = "synthetic"

# --- Legacy Path Definitions (for backward compatibility with existing inference script) ---
# These paths point to the default model's artifacts.
MODEL_NAME = MODEL_CONFIGS[DEFAULT_MODEL_NAME]["model_name"]
SCALER_NAME = MODEL_CONFIGS[DEFAULT_MODEL_NAME]["scaler_name"]
MODEL_PATH = ML_DIR / MODEL_NAME
SCALER_PATH = ML_DIR / SCALER_NAME
FEATURES = MODEL_CONFIGS[DEFAULT_MODEL_NAME]["features"]
