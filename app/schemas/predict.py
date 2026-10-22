from typing import Any

from processing.validation import CarDataInputSchema
from pydantic import BaseModel, ConfigDict


class PredictionResults(BaseModel):
    """Schema for prediction results."""

    errors: Any | None = None
    version: str
    predictions: list[float] | None = None


class MultipleCarDataInputsRequest(BaseModel):
    """Schema for multiple car prediction requests."""

    inputs: list[CarDataInputSchema]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "inputs": [
                    {
                        "brand": "Toyota",
                        "model": "Camry",
                        "model_year": 2020,
                        "milage": "50,000 mi.",
                        "fuel_type": "Gasoline",
                        "engine": "2.5L I4",
                        "transmission": "Automatic",
                        "ext_col": "White",
                        "int_col": "Black",
                        "accident": "None reported",
                        "clean_title": "Yes",
                    }
                ]
            }
        }
    )
