# --- Stage 1: Base Image ---
# We start from an official, slim Python image. 'slim' is a good choice for production
# as it contains the necessary tools without unnecessary bloat.
FROM python:3.9-slim

# --- Metadata ---
LABEL author="Gemini"
LABEL project="Smart Vehicle Log Analyzer"

# --- Environment Variables ---
# Set the working directory inside the container. All subsequent commands will run from here.
WORKDIR /app

# Set Python to run in unbuffered mode. This is recommended for Docker containers
# as it ensures that logs are sent directly to the console without being held in a buffer,
# which is crucial for real-time log monitoring.
ENV PYTHONUNBUFFERED 1

# --- Dependency Installation ---
# First, copy only the requirements.txt file.
# Docker caches layers. By copying and installing requirements separately, Docker will only
# re-run this step if the requirements.txt file changes, not on every code change.
# This significantly speeds up build times during development.
COPY ./requirements.txt .

# Install the Python dependencies.
# --no-cache-dir: Disables the pip cache, which reduces the image size.
# --upgrade pip: Ensures we have the latest version of pip.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy Application Code ---
# Copy the rest of the application source code into the working directory.
# This includes our FastAPI app, ML models, etc.
COPY ./app ./app
COPY ./data ./data

# --- Expose Port ---
# Inform Docker that the container listens on port 8000 at runtime.
# This doesn't actually publish the port, but serves as documentation.
EXPOSE 8000

# --- Command ---
# The command to run when the container starts.
# We use uvicorn to run our FastAPI application located in `app/main.py`.
# --host 0.0.0.0: Makes the server accessible from outside the container.
# --port 8000: The port to run on.
# --reload: This is useful for development, as it automatically reloads the server
# when code changes are detected. For production, you would typically remove this.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
