from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import io

# Import the main analysis service and the Pydantic models
from app.services.analysis import run_analysis
from app.api.models import AnalysisReport, HealthCheck

# An APIRouter is used to keep the API endpoints organized.
router = APIRouter()

@router.post("/analyze/", response_model=AnalysisReport, tags=["Analysis"])
async def analyze_logs_endpoint(
    file: UploadFile = File(...),
    model_name: str = Query("synthetic", enum=["synthetic", "telematics"])
):
    """
    This endpoint accepts a vehicle log file (CSV) and performs an anomaly analysis.

    - **file**: The CSV log file to be analyzed.
    - **model_name**: The analysis model to use (`synthetic` or `telematics`).
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    try:
        contents = await file.read()
        file_stream = io.BytesIO(contents)

        # Pass the model_name to the analysis service
        report = run_analysis(file_stream, file_id=file.filename, model_name=model_name)
        
        return report

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@router.get("/health/", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """
    A simple health check endpoint to confirm the API is running.
    """
    return {"status": "ok"}
