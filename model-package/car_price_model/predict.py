# validation input data
# preprocessing input data
# predict

import pandas as pd

from car_price_model import __version__ as _version
from processing.data_manager import load_preprocessing_pipeline
from processing.validation import validation_inputs

_price_pipeline = load_preprocessing_pipeline()


def make_prediction(*, input_data: pd.DataFrame) -> dict:
    """Make predictions for input car data."""

    validated_data, errors = validation_inputs(input_data=input_data)

    results = {
        "predictions": None,
        "version": _version,
        "errors": errors,
    }

    if errors is None:
        predictions = _price_pipeline.predict(validated_data)
        results["predictions"] = predictions.tolist()

    return results
