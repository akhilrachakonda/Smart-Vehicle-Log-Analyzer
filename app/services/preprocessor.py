import pandas as pd
from typing import IO
import numpy as np

# Import the centralized configuration for the feature list
from app.core.config import FEATURES, MODEL_CONFIGS

def load_and_prepare_data(file: IO) -> pd.DataFrame:
    """
    Loads data from a file-like object (e.g., uploaded file) into a pandas DataFrame,
    and performs initial validation and cleaning for the 'synthetic' data format.

    Args:
        file (IO): A file-like object containing the CSV data.

    Returns:
        pd.DataFrame: A cleaned and validated DataFrame.
    
    Raises:
        ValueError: If the file is not a valid CSV or is missing required columns.
    """
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {e}")

    required_columns = FEATURES + ['timestamp']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Uploaded log file is missing required columns for synthetic model: {missing}")

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    for col in FEATURES:
        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill()

    if df[FEATURES].isnull().sum().any():
        raise ValueError("Could not resolve all missing values in the feature columns.")

    return df

def load_telematics_data(file: IO) -> pd.DataFrame:
    """
    Safely loads data from a file-like object into a pandas DataFrame.
    Used for 'telematics' data which requires significant preprocessing.
    """
    try:
        return pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {e}")

def preprocess_telematics_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw, long-format telematics data into a clean, wide-format
    DataFrame ready for the 'telematics' model.
    """
    telematics_features = MODEL_CONFIGS['telematics']['features']
    
    # The 'variable' column in the raw data corresponds to the feature names
    # but with spaces and different casing. We create a mapping.
    # e.g., 'Vehicle speed' -> 'vehicle_speed'
    variable_map = { " ".join(word.capitalize() for word in f.split('_')): f for f in telematics_features}
    # The raw data has some inconsistencies in naming, so we add them manually
    variable_map['ENGINE RPM'] = 'engine_rpm'
    variable_map['ENGINE COOLANT TEMPERATURE'] = 'engine_coolant_temperature'
    variable_map['THROTTLE POSITION'] = 'throttle_position'
    variable_map['INTAKE MANIFOLD ABSOLUTE PRESSURE'] = 'intake_manifold_absolute_pressure'
    variable_map['AIR INTAKE TEMPERATURE'] = 'air_intake_temperature'
    variable_map['CONTROL MODULE VOLTAGE'] = 'control_module_voltage'
    variable_map['CALCULATED ENGINE LOAD'] = 'calculated_engine_load'


    selected_variables = list(variable_map.keys())

    df_filtered = df[df['variable'].isin(selected_variables)].copy()

    df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'], errors='coerce')
    df_filtered['value'] = pd.to_numeric(df_filtered['value'], errors='coerce')
    df_filtered.dropna(subset=['timestamp', 'value'], inplace=True)

    if df_filtered.empty:
        raise ValueError("The uploaded file contains no usable data for the selected telematics features.")

    df_pivot = df_filtered.pivot_table(
        index='timestamp',
        columns='variable',
        values='value',
        aggfunc='mean'
    )

    df_resampled = df_pivot.resample('1S').mean()
    df_interpolated = df_resampled.ffill().bfill()

    # Rename columns to match the feature names expected by the model
    df_final = df_interpolated.rename(columns=variable_map)
    
    # Ensure all required columns are present after processing
    if not all(f in df_final.columns for f in telematics_features):
        missing = [f for f in telematics_features if f not in df_final.columns]
        raise ValueError(f"Preprocessing failed to create all required features. Missing: {missing}")


    df_final = df_final.reset_index()

    return df_final
