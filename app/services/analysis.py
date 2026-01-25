import pandas as pd
from typing import IO, List, Dict, Any
import uuid

# --- Import Application Modules ---
# This service acts as the orchestrator, so it imports all the other components.
from app.services import preprocessor
from app.ml import inference
from app.services import interpreter
from app.api.models import AnalysisReport, Anomaly

def run_analysis(file: IO, file_id: str = None) -> AnalysisReport:
    """
    The main orchestration function for running a complete vehicle log analysis.

    Args:
        file (IO): The file-like object containing the log data.
        file_id (str, optional): A unique identifier for the file. If not provided, a new one is generated.

    Returns:
        AnalysisReport: A Pydantic model containing the full analysis report.
    """
    if file_id is None:
        file_id = str(uuid.uuid4())

    # --- Step 1: Preprocessing ---
    # Pass the raw file to the preprocessor to get a clean DataFrame.
    # Automotive Analogy: Signal acquisition and conditioning.
    print(f"[{file_id}] Starting analysis... Step 1: Preprocessing.")
    try:
        df = preprocessor.load_and_prepare_data(file)
    except ValueError as e:
        # If preprocessing fails, we return a failure report.
        return AnalysisReport(
            file_id=file_id,
            status="FAILED_PREPROCESSING",
            anomaly_count=0,
            anomalies=[]
        )

    # --- Step 2: Anomaly Detection (Inference) ---
    # Feed the clean data to the ML model to get predictions.
    # Automotive Analogy: Running the diagnostic monitor algorithm.
    print(f"[{file_id}] Step 2: Running ML inference...")
    predictions = inference.predict_anomalies(df)
    df['anomaly_flag'] = predictions

    # --- Step 3: Interpretation ---
    # Filter the DataFrame to only include the rows flagged as anomalies.
    anomalous_data = df[df['anomaly_flag'] == -1]
    print(f"[{file_id}] Step 3: Interpreting {len(anomalous_data)} detected anomalies...")

    anomaly_list: List[Anomaly] = []
    # Iterate over each anomalous row to generate an explanation.
    for _, row in anomalous_data.iterrows():
        # For each anomaly, call the interpreter to get a human-readable explanation.
        # Automotive Analogy: Mapping a raw fault condition to a DTC and its description.
        interpretation = interpreter.interpret_anomaly(row)
        
        # Create a structured Anomaly object.
        anomaly = Anomaly(
            timestamp=str(row['timestamp']),
            severity=interpretation['severity'],
            explanation=interpretation['explanation'],
            contributing_signals=row[inference.FEATURES].to_dict()
        )
        anomaly_list.append(anomaly)

    # --- Step 4: Report Generation ---
    # Assemble the final report.
    print(f"[{file_id}] Analysis complete. Generating final report.")
    report = AnalysisReport(
        file_id=file_id,
        status="COMPLETED",
        anomaly_count=len(anomaly_list),
        anomalies=anomaly_list
    )

    return report
