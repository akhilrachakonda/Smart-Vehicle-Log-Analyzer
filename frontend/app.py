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
# This is the address of our FastAPI backend.
# When running locally with Docker Compose, 'backend' is the service name.
# If running locally without Docker, you might use 'localhost' or '127.0.0.1'.
BACKEND_URL = "http://127.0.0.1:8000/api/v1/analyze/"

# --- UI Components ---
st.title("🚗 Smart Vehicle Log Analyzer")
st.markdown("""
Welcome! This tool uses a machine learning model to analyze vehicle log data, 
detect anomalies, and provide human-readable explanations for potential faults.

**How to use:**
1.  **Upload** a vehicle log file in CSV format.
2.  The system will **analyze** the data using the backend API.
3.  **Review** the generated anomaly report below.
""")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a vehicle log file (.csv)", type="csv")

if uploaded_file is not None:
    st.info(f"File uploaded: `{uploaded_file.name}`. Analyzing, please wait...")

    # --- API Request ---
    with st.spinner("Sending data to the analysis engine..."):
        try:
            # Create a dictionary for the multipart/form-data payload
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
            
            # Make the POST request to the backend
            response = requests.post(BACKEND_URL, files=files, timeout=120) # 120-second timeout

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
                    st.write("✅ No anomalies were detected in this log file. The vehicle appears to be operating normally.")
                else:
                    # Display each anomaly in an expander
                    for i, anomaly in enumerate(report['anomalies']):
                        severity = anomaly['severity']
                        color = "red" if severity == "HIGH" else "orange" if severity == "MEDIUM" else "blue"
                        
                        with st.expander(f"🚨 **{severity} Anomaly** at `{anomaly['timestamp']}`", expanded=i < 3):
                            st.markdown(f"**Explanation:**")
                            st.warning(f"{anomaly['explanation']}")
                            
                            st.markdown("**Contributing Signal Values:**")
                            # Create a dataframe for the signals for better formatting
                            signals_df = pd.DataFrame([anomaly['contributing_signals']])
                            st.dataframe(signals_df)

            else:
                # Handle API errors (e.g., 400, 500)
                st.error(f"Analysis failed. The server returned an error (Code: {response.status_code}).")
                try:
                    # Try to display the detailed error message from the API
                    error_detail = response.json().get('detail', 'No additional details provided.')
                    st.error(f"Details: {error_detail}")
                except requests.exceptions.JSONDecodeError:
                    st.error("Could not decode the error response from the server.")

        except requests.exceptions.RequestException as e:
            # Handle network errors (e.g., backend is not running)
            st.error(f"Network error: Could not connect to the backend analysis service.")
            st.error(f"Please ensure the backend is running at `{BACKEND_URL}`.")
            st.error(f"Details: {e}")
