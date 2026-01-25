# --- 0. Imports ---
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import sys

# --- 1. Path Setup ---
# Add the project root to the Python path to allow for absolute imports
# This makes the script runnable from anywhere
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "normal_vehicle_logs.csv"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "app" / "ml"
MODEL_OUTPUT_DIR.mkdir(exist_ok=True) # Ensure the directory exists

MODEL_PATH = MODEL_OUTPUT_DIR / "anomaly_model.joblib"
SCALER_PATH = MODEL_OUTPUT_DIR / "scaler.joblib"

# --- 2. Configuration ---
# These are the signals our model will learn from
# We exclude 'timestamp' (not a numeric signal for the model) and 'error_code' (our label, not a feature)
FEATURES = [
    'engine_temp', 
    'vehicle_speed',
    'battery_voltage',
    'brake_pressure'
]

# Isolation Forest parameters
# contamination='auto' is a good starting point, the model will determine the threshold
# random_state is set for reproducibility
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_samples': 'auto',
    'contamination': 'auto',
    'random_state': 42,
    'n_jobs': -1 # Use all available CPU cores
}


# --- 3. Data Loading and Preparation ---
print("Loading training data...")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Training data not found at {DATA_PATH}")
    print("Please run the 'notebooks/generate_dataset.py' script first.")
    sys.exit(1)

# Select features for training
X = df[FEATURES]

# Handle potential missing values (though our synthetic data is clean)
if X.isnull().sum().any():
    print("Missing values detected. Filling with the mean.")
    X = X.fillna(X.mean())


# --- 4. Feature Scaling ---
# Scaling is crucial for anomaly detection algorithms that are sensitive to feature ranges.
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled successfully.")


# --- 5. Model Training ---
print("Training Isolation Forest model...")
# Here, we instantiate the model with our defined parameters
model = IsolationForest(**MODEL_PARAMS)

# The 'fit' method trains the model on our clean, scaled, normal data
model.fit(X_scaled)
print("Model training complete.")


# --- 6. Save Model and Scaler ---
# We must save both the model and the scaler. 
# The scaler is required to apply the *exact same* transformation to new data during inference.
print(f"Saving model to: {MODEL_PATH}")
joblib.dump(model, MODEL_PATH)

print(f"Saving scaler to: {SCALER_PATH}")
joblib.dump(scaler, SCALER_PATH)

print("\nTraining artifacts saved successfully.")
print(f"Next step: Implement the backend service to use these artifacts for inference.")
