# --- 0. Imports ---
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import sys

# --- 1. Path Setup ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "processed_telematics_data.csv"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "app" / "ml"
MODEL_OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_OUTPUT_DIR / "telematics_anomaly_model.joblib"
SCALER_PATH = MODEL_OUTPUT_DIR / "telematics_scaler.joblib"

# --- 2. Configuration ---
# These are the signals our model will learn from.
# We use all numeric columns from the processed dataset.
FEATURES = [
    'air_intake_temperature',
    'calculated_engine_load',
    'control_module_voltage',
    'engine_coolant_temperature',
    'engine_rpm',
    'intake_manifold_absolute_pressure',
    'throttle_position',
    'vehicle_speed'
]

# Isolation Forest parameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_samples': 'auto',
    'contamination': 0.04,  # Tuned value for better precision-recall balance
    'random_state': 42,
    'n_jobs': -1
}

# --- 3. Data Loading and Preparation ---
print("Loading training data...")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Training data not found at {DATA_PATH}")
    print("Please run the 'notebooks/01_preprocess_telematics_data.py' script first.")
    sys.exit(1)

# Select features for training
# The 'timestamp' column is excluded as it's not a feature for the model.
X = df[FEATURES]

if X.isnull().sum().any():
    print("Missing values detected. Filling with the mean.")
    X = X.fillna(X.mean())

# --- 4. Feature Scaling ---
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled successfully.")

# --- 5. Model Training ---
print("Training Isolation Forest model on telematics data...")
model = IsolationForest(**MODEL_PARAMS)
model.fit(X_scaled)
print("Model training complete.")

# --- 6. Save Model and Scaler ---
print(f"Saving model to: {MODEL_PATH}")
joblib.dump(model, MODEL_PATH)

print(f"Saving scaler to: {SCALER_PATH}")
joblib.dump(scaler, SCALER_PATH)

print("\nTraining artifacts for telematics data saved successfully.")
