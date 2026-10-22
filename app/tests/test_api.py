import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    """Test that the health endpoint returns a successful response."""
    # when
    response = client.get("/api/v1/health")

    # then
    assert response.status_code == 200

    # Response Data
    response_data = response.json()

    assert response_data["name"]
    assert response_data["api_version"]
    assert response_data["model_version"]
    assert response_data["description"]


def test_make_prediction(client: TestClient, test_data: pd.DataFrame) -> None:
    """Test that the prediction endpoint returns the expected results."""

    # Given  --> prepare Test Data to JSON and Handle missing Data
    payload = {
        "inputs": test_data.replace({np.nan: None}).to_dict(
            orient="records"
        )  # Convert any row to dict and put it in list
    }

    # When
    response = client.post("/api/v1/predict", json=payload)

    # then
    assert response.status_code == 200

    prediction_data = response.json()

    assert prediction_data["errors"] is None
    assert prediction_data["version"] is not None
    assert prediction_data["predictions"]
    assert len(prediction_data["predictions"]) == len(test_data)
