import pandas as pd
from typing import IO

# Import the centralized configuration for the feature list
from app.core.config import FEATURES

def load_and_prepare_data(file: IO) -> pd.DataFrame:
    """
    Loads data from a file-like object (e.g., uploaded file) into a pandas DataFrame,
    and performs initial validation and cleaning.

    Args:
        file (IO): A file-like object containing the CSV data.

    Returns:
        pd.DataFrame: A cleaned and validated DataFrame.
    
    Raises:
        ValueError: If the file is not a valid CSV or is missing required columns.
    """
    try:
        # --- 1. Load Data ---
        # We use pandas to read the CSV data directly from the uploaded file stream.
        df = pd.read_csv(file)
    except Exception as e:
        # If pandas fails to parse the file, we raise a clear error.
        raise ValueError(f"Failed to parse CSV file: {e}")

    # --- 2. Validate Columns ---
    # Check if all the features our model needs are present in the file.
    # This is a critical safety check.
    required_columns = FEATURES + ['timestamp']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Uploaded log file is missing required columns: {missing}")

    # --- 3. Clean Data ---
    # Convert timestamp to datetime objects for potential time-based analysis
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Handle missing numerical values.
    # Here, we use forward-fill and then back-fill. This is often better for
    # time-series data than filling with the mean, as it preserves the trend.
    # Automotive Analogy: This is like the ECU holding the "Last Known Good Value"
    # for a sensor that has momentarily dropped out.
    for col in FEATURES:
        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill()

    # If any NaNs still exist (e.g., if a whole column is empty), raise an error.
    if df[FEATURES].isnull().sum().any():
        raise ValueError("Could not resolve all missing values in the feature columns.")

    return df
