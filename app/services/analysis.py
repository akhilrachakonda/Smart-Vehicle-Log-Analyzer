import pandas as pd
from typing import IO, List, Dict, Any
import uuid

# --- Import Application Modules ---
from app.services import preprocessor
from app.ml import inference
from app.services import interpreter
from app.api.models import AnalysisReport, Anomaly
from app.core.config import MODEL_CONFIGS
from app.services.llm import generate_explanation_with_llm

def run_analysis(file: IO, file_id: str = None, model_name: str = "synthetic") -> AnalysisReport:
    """
    The main orchestration function for running a complete vehicle log analysis.

    Args:
        file (IO): The file-like object containing the log data.
        file_id (str, optional): A unique identifier for the file.
        model_name (str, optional): The name of the model to use ('synthetic' or 'telematics').

    Returns:
        AnalysisReport: A Pydantic model containing the full analysis report.
    """
    if file_id is None:
        file_id = str(uuid.uuid4())

    print(f"[{file_id}] Starting analysis with model '{model_name}'... Step 1: Preprocessing.")
    
    try:
        if model_name == "telematics":
            # Telematics pipeline
            raw_df = preprocessor.load_telematics_data(file)
            df = preprocessor.preprocess_telematics_data(raw_df)
            features = MODEL_CONFIGS['telematics']['features']
        else:
            # Default synthetic pipeline
            df = preprocessor.load_and_prepare_data(file)
            features = MODEL_CONFIGS['synthetic']['features']

    except ValueError as e:
        return AnalysisReport(
            file_id=file_id,
            status="FAILED_PREPROCESSING",
            error_message=str(e),
            anomaly_count=0,
            anomalies=[]
        )

    print(f"[{file_id}] Step 2: Running ML inference...")
    if model_name == "telematics":
        predictions = inference.predict_anomalies_for_model(df, model_name="telematics")
    else:
        predictions = inference.predict_anomalies(df)
    
    df['anomaly_flag'] = predictions

    anomalous_data = df[df['anomaly_flag'] == -1]
    print(f"[{file_id}] Step 3: Interpreting {len(anomalous_data)} detected anomalies...")

    anomaly_list: List[Anomaly] = []
    for _, row in anomalous_data.iterrows():
        # The interpreter might need to be adapted if rules are different for telematics
        interpretation = interpreter.interpret_anomaly(row, feature_list=features)

        # Build context for the LLM and fetch a structured explanation.
        anomaly_context = {
            "sensor_values": row[features].to_dict(),
            "anomaly_score": float(row.get("anomaly_flag", -1)) if "anomaly_flag" in row else -1,
            "timestamp": str(row.get("timestamp")),
            "feature_name": ",".join(features),
            "rule_based_severity": interpretation["severity"],
            "rule_based_explanation": interpretation["explanation"],
        }

        llm_result = generate_explanation_with_llm(anomaly_context)
        llm_root_cause = llm_result.get("root_cause")
        llm_severity = llm_result.get("severity") or interpretation["severity"]
        llm_recommended_actions = llm_result.get("recommended_actions") or []

        final_explanation = llm_root_cause or interpretation["explanation"]
        final_severity = (llm_severity or interpretation["severity"]).upper()
        
        anomaly = Anomaly(
            timestamp=str(row['timestamp']),
            severity=final_severity,
            explanation=final_explanation,
            contributing_signals=row[features].to_dict(),
            root_cause=llm_root_cause,
            recommended_actions=llm_recommended_actions,
        )
        anomaly_list.append(anomaly)

    print(f"[{file_id}] Analysis complete. Generating final report.")
    report = AnalysisReport(
        file_id=file_id,
        status="COMPLETED",
        anomaly_count=len(anomaly_list),
        anomalies=anomaly_list
    )

    return report
