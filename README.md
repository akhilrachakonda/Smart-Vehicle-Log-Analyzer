# 🚗 Smart Vehicle Log Analyzer

## Problem Statement
Modern vehicles are complex systems generating vast amounts of log data from hundreds of sensors. Manually analyzing this data to diagnose intermittent or complex faults is inefficient and often ineffective. This project, **Smart Vehicle Log Analyzer**, provides an end-to-end system to automate this process. It ingests vehicle log files, uses a machine learning model to detect hidden anomalies, and translates these anomalies into human-readable diagnostic reports, complete with severity levels.

This system is designed as a student-friendly, production-quality application that demonstrates key concepts in automotive software, machine learning, and DevOps, inspired by professional automotive diagnostic systems like AUTOSAR's Diagnostic Event Manager (DEM).

---

## Resume-Ready Project Description
> Designed and implemented an end-to-end Smart Vehicle Log Analyzer, a cloud-native application that uses machine learning to diagnose faults in automotive sensor data. Engineered a full-stack system with a Python/FastAPI backend for analysis and a Streamlit frontend for user interaction. Deployed the application in a containerized environment using Docker and established a CI/CD pipeline with GitHub Actions for automated testing and deployment to Azure App Service. The system ingests log data, detects anomalies with an Isolation Forest model, and provides actionable, human-readable fault explanations, mirroring the functionality of an automotive ECU's diagnostic manager.

---

## System Flow
The data flows through the system in a clear, orchestrated sequence:

1.  **Upload:** A user uploads a `.csv` vehicle log file via the Streamlit web UI.
2.  **Ingestion:** The frontend sends the file to the FastAPI backend's `/analyze` endpoint.
3.  **Preprocessing:** The backend receives the file and uses a dedicated service to load, clean, and validate the data, handling missing values and ensuring correct formatting.
4.  **Inference:** The clean, scaled data is fed into the pre-trained Isolation Forest model, which flags each row as either normal (`1`) or an anomaly (`-1`).
5.  **Interpretation:** For each flagged anomaly, a rule-based "expert system" analyzes the corresponding sensor values to determine the fault's nature and assign a severity level (HIGH, MEDIUM, LOW).
6.  **Reporting:** The backend compiles the results into a structured JSON `AnalysisReport`.
7.  **Presentation:** The frontend receives the report and displays the detected anomalies and their explanations in a user-friendly dashboard.

---

## Architecture Diagram (ASCII)
This diagram shows the high-level architecture of the system, which is composed of a frontend, a backend, and a CI/CD pipeline for deployment.

```
+-----------------------------+      +-------------------------------------------------+      +------------------------+
|      User (Browser)         |      |              GitHub Repository                  |      |      Azure Cloud       |
+-----------------------------+      +-------------------------------------------------+      +------------------------+
              |                                        | (Push to main)                                |
              | 1. Upload Log File                     |                                               |
              v                                        v                                               |
+-----------------------------+      +-------------------------------------------------+      |                        |
|    Frontend (Streamlit)     |----->|              Backend (FastAPI)                  |      |                        |
| - UI for upload/display     |      | - API Endpoints (main.py, endpoints.py)         |      |                        |
+-----------------------------+      | - Analysis Service (analysis.py)                |      |                        |
              ^                      |   - Preprocessor (preprocessor.py)              |      |                        |
              | 6. Display Report    |   - ML Inference (inference.py)                 |      |                        |
              |                      |   - Fault Interpreter (interpreter.py)          |      |                        |
              |                      +-------------------------------------------------+      |                        |
              |                                        |                                               |
              | 5. Return JSON Report                  | 2. GitHub Actions CI/CD Pipeline              |
              v                                        v                                               |
+-----------------------------+      +-----------------------+  +-----------------------+      +------------------------+
|      Backend API Call       |<-----|     Build & Test      |->|   Deploy to Azure     |----->|  Azure App Service     |
| (requests)                  |      | - Linting (flake8)    |  | - Push to ACR         |      | (Running Docker Image) |
+-----------------------------+      | - Testing (pytest)    |  | - Deploy to App Svc   |      +------------------------+
                                     | - Build Docker Image  |  |                       |
                                     +-----------------------+  +-----------------------+
```

---

## How to Run Locally
You can run this project locally using Docker for a consistent, one-step setup.

**Prerequisites:**
*   Docker installed and running.
*   Git installed.

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd smart-vehicle-log-analyzer
    ```

2.  **Generate Synthetic Data & Train the Model:**
    Before building the Docker image, you need to run the data generation and training scripts locally to create the model artifacts that will be copied into the image.
    ```bash
    # Install dependencies locally for the scripts
    pip install -r requirements.txt

    # Run the scripts
    python3 notebooks/generate_dataset.py
    python3 notebooks/02_train_model.py
    ```

3.  **Build the Docker Image:**
    This command builds the Docker image for the backend application as defined in the `Dockerfile`.
    ```bash
    docker build -t smart-vehicle-analyzer-backend .
    ```

4.  **Run the Backend Container:**
    This command starts the backend server.
    ```bash
    docker run -p 8000:8000 smart-vehicle-analyzer-backend
    ```
    The backend API will now be available at `http://127.0.0.1:8000`. You can see the interactive API documentation at `http://127.0.0.1:8000/docs`.

5.  **Run the Frontend Application:**
    In a **new terminal window**, navigate to the project directory and run the Streamlit app.
    ```bash
    # Ensure you are in the project's root directory
    streamlit run frontend/app.py
    ```
    The frontend will be available at `http://localhost:8501`. You can now open this address in your browser to use the application.

---

## How to Deploy on Azure (Free Tier)
This guide explains how to deploy the containerized backend to Azure App Service.

**Prerequisites:**
*   An Azure account (Free Tier is sufficient).
*   Azure CLI installed and configured (`az login`).
*   A GitHub repository with your project code.

**Step-by-Step Guide:**

1.  **Create an Azure Container Registry (ACR):**
    ACR is a private registry for your Docker images.
    ```bash
    az group create --name vehicle-analyzer-rg --location "East US"
    az acr create --resource-group vehicle-analyzer-rg --name <your-unique-acr-name> --sku Basic --admin-enabled true
    ```

2.  **Set up GitHub Actions Secrets:**
    In your GitHub repository, go to `Settings > Secrets and variables > Actions` and add the following secrets. These allow GitHub Actions to log in to your Azure account.
    *   `AZURE_CREDENTIALS`: The JSON output of `az ad sp create-for-rbac --name "github-actions" --role contributor --scopes /subscriptions/<your-subscription-id> --sdk-auth`.
    *   `ACR_LOGIN_SERVER`: The login server of your ACR (e.g., `<your-unique-acr-name>.azurecr.io`).
    *   `ACR_USERNAME`: The username for your ACR (get with `az acr credential show ...`).
    *   `ACR_PASSWORD`: The password for your ACR.

3.  **Create the GitHub Actions Workflow:**
    Create a file at `.github/workflows/deploy.yml`. This workflow will trigger on a push to the `main` branch, build the Docker image, push it to your ACR, and deploy it to App Service. (A sample workflow file should be included in the project).

4.  **Create an Azure App Service Plan:**
    This defines the underlying VM for your web app. The F1 (Free) tier is sufficient.
    ```bash
    az appservice plan create --name vehicle-analyzer-plan --resource-group vehicle-analyzer-rg --sku F1 --is-linux
    ```

5.  **Create the Web App (App Service):**
    This creates the App Service instance that will run your container.
    ```bash
    az webapp create --resource-group vehicle-analyzer-rg --plan vehicle-analyzer-plan --name <your-unique-app-name> --deployment-container-image-name <your-acr-login-server>/smart-vehicle-analyzer-backend:latest
    ```

6.  **Enable Continuous Deployment:**
    Configure the App Service to automatically pull new images when they are pushed to ACR.
    ```bash
    az webapp deployment container config --enable-cd true --name <your-unique-app-name> --resource-group vehicle-analyzer-rg
    ```

7.  **Push to GitHub:**
    Commit and push your code to the `main` branch. This will trigger the GitHub Actions workflow, which will build and deploy your application automatically. After the workflow completes, your application will be live at `http://<your-unique-app-name>.azurewebsites.net`.
