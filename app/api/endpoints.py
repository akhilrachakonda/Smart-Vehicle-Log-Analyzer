from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io

# Import the main analysis service and the Pydantic models
from app.services.analysis import run_analysis
from app.api.models import AnalysisReport, HealthCheck

# An APIRouter is used to keep the API endpoints organized.
# This is good practice for larger applications.
router = APIRouter()

@router.post("/analyze/", response_model=AnalysisReport, tags=["Analysis"])
async def analyze_logs_endpoint(file: UploadFile = File(...)):
    """
    This endpoint accepts a vehicle log file (CSV) and performs an anomaly analysis.

    - **file**: The CSV log file to be analyzed.
    """
    # --- 1. File Validation ---
    # Basic check to ensure the uploaded file is a CSV.
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    try:
        # --- 2. Read File Content ---
        # The UploadFile object is read into an in-memory bytes buffer.
        # This is efficient as it avoids saving the file to disk on the server.
        contents = await file.read()
        file_stream = io.BytesIO(contents)

        # --- 3. Call Analysis Service ---
        # The file stream is passed to our main analysis service.
        # This is where the core logic of the application is executed.
        report = run_analysis(file_stream, file_id=file.filename)
        
        # --- 4. Return Report ---
        # The final report is returned as a JSON response.
        # FastAPI automatically serializes the Pydantic model to JSON.
        return report

    except ValueError as e:
        # Catch specific ValueErrors raised by our services (e.g., missing columns)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # A general catch-all for any other unexpected errors during analysis.
        # In a production system, this would log the full error for debugging.
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@router.get("/health/", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """
    A simple health check endpoint to confirm the API is running.
    """
    return {"status": "ok"}
