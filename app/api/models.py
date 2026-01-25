from pydantic import BaseModel
from typing import List, Dict, Any

# --- Anomaly Data Model ---
# This model defines the structure for a single detected anomaly.
# It ensures that every anomaly report we generate is consistent.
# Automotive Analogy: This is like the structure of a single Diagnostic Trouble Code (DTC)
# record, which contains the fault code, its status, and related data.
class Anomaly(BaseModel):
    timestamp: str
    severity: str
    explanation: str
    contributing_signals: Dict[str, Any]

# --- Analysis Report Model ---
# This is the main response model for the /analyze endpoint.
# It provides a summary of the analysis and a list of all found anomalies.
class AnalysisReport(BaseModel):
    file_id: str
    status: str
    anomaly_count: int
    anomalies: List[Anomaly]

# --- Health Check Model ---
# A simple model for the health check endpoint.
class HealthCheck(BaseModel):
    status: str
