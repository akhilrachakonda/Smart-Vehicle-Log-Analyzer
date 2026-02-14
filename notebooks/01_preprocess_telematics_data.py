
import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
# Select a subset of variables to act as our core features.
# These are chosen based on their frequency and relevance to vehicle performance.
SELECTED_VARIABLES = [
    'Vehicle speed',
    'ENGINE RPM',
    'ENGINE COOLANT TEMPERATURE',
    'THROTTLE POSITION',
    'INTAKE MANIFOLD ABSOLUTE PRESSURE',
    'AIR INTAKE TEMPERATURE',
    'CONTROL MODULE VOLTAGE',
    'CALCULATED ENGINE LOAD',
]

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_PATH = DATA_DIR / "Telematicsdata.csv"
OUTPUT_PATH = DATA_DIR / "processed_telematics_data.csv"

# --- Main Preprocessing Logic ---
if __name__ == "__main__":
    print(f"Loading raw data from {INPUT_PATH}...")
    try:
        df = pd.read_csv(INPUT_PATH)

        # --- 1. Initial Filtering and Cleaning ---
        print("Filtering for selected variables...")
        df_filtered = df[df['variable'].isin(SELECTED_VARIABLES)].copy()

        # Convert 'timestamp' to datetime objects for time-series analysis
        df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'], errors='coerce')

        # Convert 'value' to numeric, coercing non-numeric values to NaN (Not a Number)
        df_filtered['value'] = pd.to_numeric(df_filtered['value'], errors='coerce')

        # Drop rows where timestamp or value could not be parsed
        df_filtered.dropna(subset=['timestamp', 'value'], inplace=True)

        print(f"Filtered data has {len(df_filtered)} rows.")

        # --- 2. Pivot Data from Long to Wide Format ---
        print("Pivoting data to wide format...")
        # We use a pivot table to turn unique 'variable' names into columns.
        # The index is the timestamp, and values are the sensor readings.
        df_pivot = df_filtered.pivot_table(
            index='timestamp',
            columns='variable',
            values='value',
            aggfunc='mean' # Use mean to handle multiple readings in the same timestamp
        )

        print(f"Pivot table created with {len(df_pivot)} rows.")

        # --- 3. Resample and Interpolate ---
        # The pivot table will have irregular time intervals. We need to create a uniform time index.
        print("Resampling data to a consistent 1-second frequency...")
        
        # Resample the data to have one row per second.
        # Use 'mean()' to aggregate any data points that fall into the same second.
        df_resampled = df_pivot.resample('1S').mean()

        print(f"Resampled data has {len(df_resampled)} rows.")
        
        # Interpolate missing values. After resampling, many rows will be empty (NaN).
        # 'ffill' (forward fill) propagates the last valid observation forward.
        # 'bfill' (backward fill) fills remaining NaNs from the next valid observation.
        # This ensures we have a continuous dataset without gaps.
        print("Interpolating missing values...")
        df_interpolated = df_resampled.ffill().bfill()

        # --- 4. Final Cleanup and Save ---
        # Rename columns to be more script-friendly (e.g., remove spaces)
        df_interpolated.columns = [
            col.replace(' ', '_').lower() for col in df_interpolated.columns
        ]

        # Reset index to turn the 'timestamp' index back into a column
        df_final = df_interpolated.reset_index()

        print(f"Saving processed data to {OUTPUT_PATH}...")
        df_final.to_csv(OUTPUT_PATH, index=False)

        print("\nPreprocessing complete.")
        print("\n--- Processed Data Sample ---")
        print(df_final.head())
        print("\n--- Processed Data Info ---")
        df_final.info()

    except FileNotFoundError:
        print(f"Error: The file {INPUT_PATH} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
