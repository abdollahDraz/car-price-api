# Car Price Prediction API

A machine learning project that predicts used car prices through a REST API built with FastAPI. The project includes a reusable model package, automated testing, Docker containerization, and a CI/CD deployment workflow.

## Live Demo

- **API:** https://car-price-api.abdmahdra.blitz.cloud/
- **API Documentation (Swagger UI):** https://car-price-api.abdmahdra.blitz.cloud/docs
- **Health Check:** https://car-price-api.abdmahdra.blitz.cloud/api/v1/health

## Project Overview

This project demonstrates an end-to-end workflow for deploying a machine learning model as a REST API.

The trained model is packaged separately so it can be reused independently of the API. FastAPI exposes prediction and health-check endpoints, while automated tests help verify API behavior and detect unexpected changes in model predictions.

## Architecture

```text
Model Development
       |
       v
Reusable Model Package
       |
       v
Package Distribution
       |
       v
FastAPI REST API
       |
       v
Automated Testing
       |
       v
Docker Image
       |
       v
CI/CD Pipeline
       |
       v
Cloud Deployment
```

## Tech Stack

- **Python** â€” core programming language
- **Pandas and NumPy** â€” data processing
- **Scikit-learn** â€” machine learning pipeline
- **FastAPI** â€” REST API framework
- **Pydantic** â€” request validation and schemas
- **Uvicorn** â€” ASGI server
- **Pytest** â€” automated testing
- **Tox** â€” test orchestration
- **Differential Testing** â€” detecting unexpected changes in model predictions
- **Bump My Version** â€” API version management
- **Docker** â€” containerization
- **CircleCI** â€” continuous integration and delivery
- **Docker Hub** â€” container image registry
- **Blitz** â€” cloud deployment platform

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome page |
| GET | `/api/v1/health` | Returns API and model version information |
| POST | `/api/v1/predict` | Predicts used car prices from input data |

### Interactive API Documentation

Visit the `/docs` endpoint on the live API to inspect the available endpoints and submit requests through Swagger UI.

## Repository Structure

```text
car-price-api/
â”œâ”€â”€ app/
â”‚   â”œâ”€â”€ api.py
â”‚   â”œâ”€â”€ config.py
â”‚   â”œâ”€â”€ main.py
â”‚   â”œâ”€â”€ schemas/
â”‚   â””â”€â”€ tests/
â”œâ”€â”€ datasets/
â”œâ”€â”€ model-package/
â”œâ”€â”€ .circleci/
â”œâ”€â”€ Dockerfile
â”œâ”€â”€ Procfile
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ test_requirements.txt
â”œâ”€â”€ tox.ini
â””â”€â”€ README.md
```

*The tree highlights the main project components. Some files and directories may be omitted.*

## Model Package

The machine learning pipeline is maintained as a reusable Python package, separate from the API application.

This separation keeps model preprocessing and prediction logic independent of the web-serving layer.

## Testing

The project includes API tests and differential tests.

Run these commands from the project root in an environment where the project dependencies are installed.

Run API tests through Tox:

```powershell
python -m tox run -e test_app
```

Run differential tests:

```powershell
python -m pytest -m differential
```

Run the configured Tox environments:

```powershell
python -m tox
```

## Docker

Build the API image from the project root:

```powershell
docker build -t car-price-api .
```

Run the container locally:

```powershell
docker run --rm -p 8000:8000 car-price-api
```

If the Dockerfile requires a specific model-package version, provide the build argument expected by the current Dockerfile.

After starting the container, open:

http://localhost:8000/docs

## CI/CD Workflow

The project uses CircleCI to automate the deployment workflow.

The pipeline includes the following stages:

1. Building or preparing the model package.
2. Testing the API.
3. Building and publishing the Docker image.
4. Triggering the configured deployment workflow.

The exact execution and deployment behavior depends on the repository's CircleCI configuration and environment variables.

## Versioning

The API version and model-package version are tracked separately. This makes it easier to identify whether a change belongs to the serving application or the machine learning package.

## Author

**Abdollah Draz**

- GitHub: [abdollahDraz](https://github.com/abdollahDraz)

