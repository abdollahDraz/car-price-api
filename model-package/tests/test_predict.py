# testing predictions

import math

import pandas as pd

from car_price_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    """Testing make prediction"""
    # Given
    expected_first_prediction_value = 15351.093333333334
    expected_no_predictions = len(sample_input_data)

    # when
    result = make_prediction(
        input_data=sample_input_data
    )  # predictions & errors & version

    # then
    predictions = result.get("predictions")

    assert isinstance(predictions, list)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert math.isclose(predictions[0], expected_first_prediction_value, abs_tol=100)


def test_make_prediction_with_missing_feature(sample_input_data):
    """Test make prediction with missing feature"""
    # Given
    invalid_input_data = sample_input_data.drop(columns=["brand"])

    # when
    result = make_prediction(input_data=invalid_input_data)

    # then
    assert result["predictions"] is None
    assert result["errors"] is not None


def test_make_prediction_with_invalid_value(sample_input_data):
    """Verify that make_prediction raises an error when a feature
    contains an invalid value.
    """
    # Given
    invalid_input_data = sample_input_data.copy()
    invalid_input_data["model_year"] = invalid_input_data["model_year"].astype(object)

    invalid_input_data.loc[0, "model_year"] = "not-a-year"

    # when
    results = make_prediction(input_data=invalid_input_data)

    # then
    assert results["predictions"] is None
    assert results["errors"] is not None


def test_make_prediction_with_multiple_inputs(sample_input_data):
    """Verify that make_prediction returns one prediction per input row"""

    # when
    multiple_input_data = pd.concat(
        [sample_input_data, sample_input_data], ignore_index=True
    )
    # when
    result = make_prediction(input_data=multiple_input_data)

    # then
    assert result["errors"] is None
    assert isinstance(result["predictions"], list)
    assert len(result["predictions"]) == 2
