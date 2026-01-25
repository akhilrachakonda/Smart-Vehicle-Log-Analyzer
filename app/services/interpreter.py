import pandas as pd
from typing import Dict, Any

# --- Fault Interpretation Rules ---
# This is the core of our "expert system". We define rules that check for specific,
# physically meaningful conditions in the data that correspond to known fault types.
# Automotive Analogy: This is the logic a diagnostic engineer would program into an ECU
# to set a specific DTC. For example, "IF engine_speed > 2000 RPM AND throttle_angle < 5%
# AND manifold_pressure is HIGH, THEN set fault 'P0106'".

def interpret_anomaly(row: pd.Series) -> Dict[str, Any]:
    """
    Interprets a single anomalous data row and returns a structured explanation.

    Args:
        row (pd.Series): A row of data corresponding to a detected anomaly.

    Returns:
        Dict[str, Any]: A dictionary containing the explanation and severity.
    """
    
    # --- Rule 1: Engine Overheating Fault ---
    # Condition: High engine temperature, especially when the vehicle is slow or idle.
    # This points to a cooling system failure (e.g., stuck thermostat, fan failure).
    if row['engine_temp'] > 105 and row['vehicle_speed'] < 20:
        return {
            "severity": "HIGH",
            "explanation": "Critical Overheating Detected: Engine temperature is dangerously high at low vehicle speed. Suspect cooling system failure. Immediate attention required."
        }
    if row['engine_temp'] > 100 and row['vehicle_speed'] < 40:
        return {
            "severity": "MEDIUM",
            "explanation": "Engine Temperature Anomaly: Engine is running hotter than normal for the current vehicle speed. Potential cooling issue."
        }

    # --- Rule 2: Alternator/Battery Failure ---
    # Condition: Battery voltage is abnormally low, especially when the engine is running.
    # This suggests the alternator is not charging the battery correctly.
    if row['battery_voltage'] < 12.0 and row['vehicle_speed'] > 10:
        return {
            "severity": "HIGH",
            "explanation": "Charging System Fault: Battery voltage is critically low while the vehicle is in motion. Suspect alternator failure. Risk of vehicle stalling."
        }
    if row['battery_voltage'] < 12.2:
        return {
            "severity": "MEDIUM",
            "explanation": "Low Battery Voltage: Battery voltage is below the normal operating range. Could indicate an aging battery or early-stage alternator issue."
        }

    # --- Rule 3: Unintended Braking / Brake System Fault ---
    # Condition: High brake pressure is applied when the vehicle is at high speed
    # without a corresponding decrease in speed. This is highly anomalous.
    if row['brake_pressure'] > 50 and row['vehicle_speed'] > 80:
        return {
            "severity": "HIGH",
            "explanation": "Brake System Anomaly: High brake pressure detected at highway speeds without significant deceleration. This is highly unusual and could indicate a sensor fault or unintended braking."
        }

    # --- Generic / Fallback Rule ---
    # If no specific rule matches, we provide a generic explanation.
    # This is important for catching unexpected anomalies the model finds.
    return {
        "severity": "LOW",
        "explanation": "General Anomaly Detected: The model identified an unusual combination of sensor readings. While not matching a specific critical fault, this pattern deviates from normal operation."
    }
