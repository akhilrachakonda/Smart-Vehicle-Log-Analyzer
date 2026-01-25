from fastapi import FastAPI
from app.api.endpoints import router as api_router

# --- Application Initialization ---
# Create the main FastAPI application instance.
# The title and version will be used in the auto-generated API documentation.
app = FastAPI(
    title="Smart Vehicle Log Analyzer",
    version="1.0.0",
    description="An API to analyze vehicle log data for anomalies using machine learning."
)

# --- Include API Routes ---
# The router from our endpoints module is included here.
# This connects the endpoint definitions (e.g., @router.post("/analyze"))
# to the main application, making them accessible.
app.include_router(api_router, prefix="/api/v1")

# --- Root Endpoint ---
# A simple endpoint at the root URL to welcome users.
@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": "Welcome to the Smart Vehicle Log Analyzer API.",
        "documentation": "/docs"
    }

# The following is not strictly necessary for the app to run, but it's good practice
# to confirm that the model loading (which happens on module import) was successful at startup.
# This is a simple "self-test".
from app.ml.inference import get_model_artifacts

@app.on_event("startup")
async def startup_event():
    """
    This event is triggered when the application starts.
    We use it to explicitly check if the ML artifacts are loaded.
    """
    print("Application startup...")
    try:
        get_model_artifacts()
        print("ML model and scaler artifacts loaded successfully.")
    except RuntimeError as e:
        print(f"FATAL ERROR: {e}")
        # In a real production scenario, you might want to prevent the app
        # from starting if the models can't be loaded.
        # For this project, we'll print a clear error.
    print("Startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    """
    This event is triggered when the application shuts down.
    """
    print("Application shutdown.")
