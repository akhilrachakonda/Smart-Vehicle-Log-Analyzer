import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# --- Configuration ---
N_SAMPLES_NORMAL = 5000
N_SAMPLES_MIXED = 2000

# --- Path Setup ---
# Make paths robust by defining them relative to the script's location
# This ensures the script can be run from anywhere
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True) # Create the data directory if it doesn't exist

OUTPUT_NORMAL_PATH = DATA_DIR / "normal_vehicle_logs.csv"
OUTPUT_MIXED_PATH = DATA_DIR / "mixed_vehicle_logs.csv"

# --- Signal Characteristics ---
# Normal Operating Ranges
TEMP_NORMAL = (85, 100)
SPEED_NORMAL = (0, 120)
VOLTAGE_NORMAL = (12.2, 14.4)
BRAKE_NORMAL = (0, 20) # Normal, light braking

# --- Function to Generate Base Data ---
def generate_base_data(n_samples):
    """Generates a base DataFrame with realistic signal correlations."""
    data = pd.DataFrame()
    
    # Create a time index
    now = datetime.now()
    data['timestamp'] = [now + timedelta(seconds=i) for i in range(n_samples)]
    
    # Simulate different driving phases
    phases = np.random.choice(['idle', 'city', 'highway'], n_samples, p=[0.2, 0.5, 0.3])
    
    # --- Generate Signals based on Phases ---
    # Vehicle Speed
    speed = np.zeros(n_samples)
    speed[phases == 'idle'] = np.random.uniform(0, 1, size=(phases == 'idle').sum())
    speed[phases == 'city'] = np.random.uniform(10, 60, size=(phases == 'city').sum())
    speed[phases == 'highway'] = np.random.uniform(80, 120, size=(phases == 'highway').sum())
    data['vehicle_speed'] = np.clip(speed + np.random.normal(0, 2, n_samples), 0, 200)

    # Engine Temperature (correlates with speed, but with lag)
    temp = np.full(n_samples, 80.0)
    for i in range(1, n_samples):
        if speed[i] > 60:
            temp[i] = temp[i-1] + np.random.uniform(0.1, 0.5) # Heats up on highway
        elif speed[i] < 10:
            temp[i] = temp[i-1] - np.random.uniform(0.05, 0.2) # Cools down at idle
        else:
            temp[i] = temp[i-1] + np.random.uniform(-0.1, 0.1) # Stable in city
    data['engine_temp'] = np.clip(temp + np.random.normal(0, 1, n_samples), 70, 120)

    # Battery Voltage
    voltage = np.random.uniform(VOLTAGE_NORMAL[0], VOLTAGE_NORMAL[1], n_samples)
    # Slight drop when speed is high
    voltage[speed > 80] -= np.random.uniform(0.1, 0.3)
    data['battery_voltage'] = np.clip(voltage, 11.5, 14.8)

    # Brake Pressure
    brake = np.zeros(n_samples)
    brake[phases == 'city'] = np.random.exponential(scale=5, size=(phases == 'city').sum())
    data['brake_pressure'] = np.clip(brake, 0, 100)
    
    # Error Code (0 means no error)
    data['error_code'] = 0
    
    return data

# --- Function to Inject Anomalies ---
def inject_anomalies(df):
    """Injects specific, meaningful anomalies into the dataframe."""
    df_anomalous = df.copy()
    
    # Anomaly 1: Overheating at low speed (Thermostat Stuck)
    # High engine temp while vehicle speed is low.
    anomaly_idx_1 = df_anomalous[(df_anomalous['vehicle_speed'] < 10) & (df_anomalous.index > 100)].sample(frac=0.1).index
    df_anomalous.loc[anomaly_idx_1, 'engine_temp'] = np.random.uniform(105, 115, len(anomaly_idx_1))
    df_anomalous.loc[anomaly_idx_1, 'error_code'] = 101 # Custom code for this fault

    # Anomaly 2: Alternator Failure
    # Battery voltage drops significantly, especially at speed.
    anomaly_idx_2 = df_anomalous[df_anomalous['vehicle_speed'] > 40].sample(frac=0.05).index
    df_anomalous.loc[anomaly_idx_2, 'battery_voltage'] = np.random.uniform(10.5, 11.8, len(anomaly_idx_2))
    df_anomalous.loc[anomaly_idx_2, 'error_code'] = 202 # Custom code

    # Anomaly 3: Brake System Fault
    # Brake pressure is high even when speed is high (unintended braking).
    anomaly_idx_3 = df_anomalous[df_anomalous['vehicle_speed'] > 80].sample(frac=0.05).index
    df_anomalous.loc[anomaly_idx_3, 'brake_pressure'] = np.random.uniform(50, 80, len(anomaly_idx_3))
    df_anomalous.loc[anomaly_idx_3, 'error_code'] = 305 # Custom code
    
    return df_anomalous

# --- Main Generation Logic ---
if __name__ == "__main__":
    print("Generating synthetic vehicle log data...")
    
    # 1. Generate Normal Data for Training
    normal_data = generate_base_data(N_SAMPLES_NORMAL)
    normal_data.to_csv(OUTPUT_NORMAL_PATH, index=False)
    print(f"Successfully generated {N_SAMPLES_NORMAL} normal samples at: {OUTPUT_NORMAL_PATH}")

    # 2. Generate Mixed Data for Testing/Inference
    mixed_data_base = generate_base_data(N_SAMPLES_MIXED)
    mixed_data_anomalous = inject_anomalies(mixed_data_base)
    mixed_data_anomalous.to_csv(OUTPUT_MIXED_PATH, index=False)
    print(f"Successfully generated {N_SAMPLES_MIXED} mixed (normal + anomaly) samples at: {OUTPUT_MIXED_PATH}")
    
    print("\nData generation complete.")
    print("\n--- Normal Data Sample ---")
    print(normal_data.head())
    print("\n--- Mixed Data Sample (check for anomalies) ---")
    print(mixed_data_anomalous[mixed_data_anomalous['error_code'] != 0].head())
