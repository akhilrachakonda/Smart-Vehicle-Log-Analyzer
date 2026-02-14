import streamlit as st
import requests
import pandas as pd
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Vehicle Log Analyzer",
    page_icon="🚗",
    layout="wide"
)

# --- Backend API Configuration ---
BACKEND_URL = "http://127.0.0.1:8000/api/v1/analyze/"

# --- UI Components ---
st.title("🚗 Smart Vehicle Log Analyzer")
st.markdown("""
Welcome! This tool uses a machine learning model to analyze vehicle log data, 
detect anomalies, and provide human-readable explanations for potential faults.

**How to use:**
1.  **Select the model** appropriate for your data type.
2.  **Upload** a vehicle log file in CSV format.
3.  The system will **analyze** the data.
4.  **Review** the generated anomaly report below.
""")

# --- Model Selection ---
model_choice = st.selectbox(
    "1. Select the Analysis Model",
    ("synthetic", "telematics"),
    help="Choose 'synthetic' for generated data, or 'telematics' for raw, long-format telematics data."
)

# --- File Uploader ---
uploaded_file = st.file_uploader("2. Choose a vehicle log file (.csv)", type="csv")

if uploaded_file is not None:
    st.info(f"File uploaded: `{uploaded_file.name}`. Analyzing with '{model_choice}' model, please wait...")

    # --- API Request ---
    with st.spinner("Sending data to the analysis engine..."):
        try:
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
            
            # Append the selected model name as a query parameter
            analysis_url = f"{BACKEND_URL}?model_name={model_choice}"
            
            response = requests.post(analysis_url, files=files, timeout=120)

            # --- Response Handling ---
            if response.status_code == 200:
                st.success("Analysis complete!")
                report = response.json()

                # --- Display Report ---
                st.header("Analysis Report")
                
                col1, col2 = st.columns(2)
                col1.metric("File Analyzed", report.get('file_id', 'N/A'))
                col2.metric("Total Anomalies Detected", report.get('anomaly_count', 0))

                st.subheader("Detected Anomalies")
                if not report['anomalies']:
                    st.write("✅ No anomalies were detected in this log file.")
                else:
                    for i, anomaly in enumerate(report['anomalies']):
                        severity = anomaly['severity']
                        severity_normalized = severity.upper()
                        color = "red" if severity_normalized == "HIGH" else "orange" if severity_normalized == "MEDIUM" else "blue"
                        
                        with st.expander(f"🚨 **{severity_normalized} Anomaly** at `{anomaly['timestamp']}`", expanded=i < 3):
                            root_cause = anomaly.get('root_cause') or anomaly.get('explanation')
                            st.markdown("**Explanation:**")
                            st.warning(f"{root_cause}")
                            
                            actions = anomaly.get('recommended_actions') or []
                            if actions:
                                st.markdown("**Recommended Actions:**")
                                for action in actions:
                                    st.write(f"- {action}")
                            
                            st.markdown("**Contributing Signal Values:**")
                            signals_df = pd.DataFrame([anomaly['contributing_signals']])
                            st.dataframe(signals_df)

            else:
                st.error(f"Analysis failed (Code: {response.status_code}).")
                try:
                    error_detail = response.json().get('detail', 'No details provided.')
                    st.error(f"Details: {error_detail}")
                except requests.exceptions.JSONDecodeError:
                    st.error("Could not decode server error response.")

        except requests.exceptions.RequestException as e:
            st.error(f"Network error: Could not connect to the backend at `{BACKEND_URL}`.")
            st.error(f"Details: {e}")
