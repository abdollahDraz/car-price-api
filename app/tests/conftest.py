from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import DATASET_PATH
from app.main import app


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    """Load test data for API tests."""
    return pd.read_csv(DATASET_PATH)


@pytest.fixture(scope="module")
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client  # type of Client is : TestClient
        app.dependency_overrides = {}
