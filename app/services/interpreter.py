import pandas as pd
from typing import Dict, Any, List

def interpret_anomaly(row: pd.Series, feature_list: List[str]) -> Dict[str, Any]:
    """
    Interprets a single anomalous data row and returns a structured explanation.

    Args:
        row (pd.Series): A row of data corresponding to a detected anomaly.
        feature_list (List[str]): The list of features available in the row.

    Returns:
        Dict[str, Any]: A dictionary containing the explanation and severity.
    """
    
    # Define a helper to check for a feature's existence and get its value
    def get_feature(name_options):
        for name in name_options:
            if name in feature_list:
                return row[name]
        return None

    engine_temp = get_feature(['engine_temp', 'engine_coolant_temperature'])
    vehicle_speed = get_feature(['vehicle_speed'])
    battery_voltage = get_feature(['battery_voltage', 'control_module_voltage'])
    brake_pressure = get_feature(['brake_pressure'])

    # --- Rule 1: Engine Overheating Fault ---
    if engine_temp is not None and vehicle_speed is not None:
        if engine_temp > 105 and vehicle_speed < 20:
            return {
                "severity": "HIGH",
                "explanation": "Critical Overheating Detected: Engine temperature is dangerously high at low vehicle speed. Suspect cooling system failure."
            }
        if engine_temp > 100 and vehicle_speed < 40:
            return {
                "severity": "MEDIUM",
                "explanation": "Engine Temperature Anomaly: Engine is running hotter than normal for the current vehicle speed."
            }

    # --- Rule 2: Alternator/Battery Failure ---
    if battery_voltage is not None and vehicle_speed is not None:
        if battery_voltage < 12.0 and vehicle_speed > 10:
            return {
                "severity": "HIGH",
                "explanation": "Charging System Fault: Battery voltage is critically low while the vehicle is in motion. Suspect alternator failure."
            }
        if battery_voltage < 12.2:
            return {
                "severity": "MEDIUM",
                "explanation": "Low Battery Voltage: Battery voltage is below the normal operating range."
            }

    # --- Rule 3: Unintended Braking / Brake System Fault ---
    if brake_pressure is not None and vehicle_speed is not None:
        if brake_pressure > 50 and vehicle_speed > 80:
            return {
                "severity": "HIGH",
                "explanation": "Brake System Anomaly: High brake pressure detected at highway speeds without significant deceleration."
            }

    # --- Generic / Fallback Rule ---
    return {
        "severity": "LOW",
        "explanation": "General Anomaly Detected: The model identified an unusual combination of sensor readings that deviates from normal operation."
    }
