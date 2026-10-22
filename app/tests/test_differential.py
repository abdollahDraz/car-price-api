
import math

import pandas as pd
import pytest

from app.config import DATASET_PATH
from car_price_model.predict import make_prediction
from config.config import TARGET


@pytest.mark.differential
def test_model_prediction_differential() -> None:
    """
    Compare current model predictions with the saved baseline.
    """

    # Load baseline predictions
    baseline_path = DATASET_PATH.parent / "test_data_predictions.csv"
    previous_model_df = pd.read_csv(baseline_path)
    previous_predictions = previous_model_df["predictions"].tolist()   

    # Prepare test input data
    test_data = pd.read_csv(DATASET_PATH)
    input_data = test_data.drop(columns=[TARGET])
    input_data = input_data.astype(object).where(
        pd.notna(input_data), None
    )

    # Get current model predictions
    current_result = make_prediction(input_data=input_data)

    if current_result["errors"] is not None:
        pytest.fail(current_result["errors"])

    current_predictions = current_result["predictions"]

    # Check prediction counts
    assert len(previous_predictions) == len(current_predictions)

    # Compare previous and current predictions
    for previous_value, current_value in zip(
        previous_predictions, current_predictions
    ):
        assert math.isclose(
            previous_value,
            current_value,
            rel_tol=0.05,
        ), (
            f"Prediction changed beyond tolerance: "
            f"previous={previous_value}, current={current_value}"
        )
