# for check up the input data

import warnings
from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from config.config import config
from processing.features import PriceTransformer


class CarDataInputSchema(BaseModel):
    """Schema for validating a single raw car input"""

    model_config = ConfigDict(extra="ignore")  # ignore the extra input features
    brand: Optional[str] = None
    model: Optional[str] = None
    model_year: Optional[float] = None
    milage: Optional[str | float] = None
    fuel_type: Optional[str] = None
    engine: Optional[str] = None
    transmission: Optional[str] = None
    ext_col: Optional[str] = None
    int_col: Optional[str] = None
    accident: Optional[str] = None
    clean_title: Optional[str] = None


class MultipleCarDataInputs(BaseModel):
    """Schema for validating multiple car inputs"""

    inputs: list[CarDataInputSchema]


def validate_features(
    *, input_data: pd.DataFrame, expected_features: list[str]
) -> Optional[str]:
    """Validate required feature columns"""

    # case : Duplicated features
    if input_data.columns.duplicated().any():
        duplicated_columns = input_data.columns[
            input_data.columns.duplicated()
        ].to_list()
        return f"Duplicate columns found: {duplicated_columns}"

    # storing missing features
    missing_features = [
        feature for feature in expected_features if feature not in input_data.columns
    ]

    # case : missing features
    if missing_features:
        return f"Missing required features: {missing_features}"

    return None


# for inference
def validation_inputs(
    *,
    input_data: pd.DataFrame,
) -> tuple[pd.DataFrame, Optional[str]]:
    """Validate the input data before making predictions."""

    expected_features = list(CarDataInputSchema.model_fields.keys())

    error = validate_features(
        input_data=input_data,
        expected_features=expected_features,
    )

    if error is not None:
        return input_data, error

    validated_data = input_data.loc[:, expected_features].copy()

    try:
        validated_records = MultipleCarDataInputs(
            inputs=validated_data.to_dict(orient="records")
        )

        validated_data = pd.DataFrame(
            [record.model_dump() for record in validated_records.inputs],
            columns=expected_features,
        )

    except ValidationError as error:
        return input_data, str(error)

    return validated_data, None


def validate_training_data(
    *, input_data: pd.DataFrame
) -> tuple[pd.DataFrame, Optional[str]]:
    """Validate the dataset before model training"""

    # expected Features & target feature  , it reference
    expected_features = list(CarDataInputSchema.model_fields.keys())
    target = config.model_settings.target

    # validate required feature columns
    error = validate_features(
        input_data=input_data, expected_features=expected_features
    )
    if error is not None:
        return input_data, error

    # validate target column
    error = validate_features(input_data=input_data, expected_features=[target])
    if error is not None:
        return input_data, error

    # prepare validated data
    required_columns = expected_features + [target]
    validated_data = input_data.loc[:, required_columns].copy()

    # loading target transformer
    target_transformer = PriceTransformer(variables=[target])
    cleaned_target = target_transformer.fit_transform(validated_data[[target]])[target]

    valid_rows = cleaned_target.notna()
    excluded_rows = int((~valid_rows).sum())

    if excluded_rows:
        warnings.warn(
            f"Excluded {excluded_rows} rows with missing or invalid "
            f"target values in '{target}'",
            UserWarning,
            stacklevel=2,
        )

    validated_data = validated_data.loc[valid_rows].copy()
    validated_data[target] = cleaned_target.loc[valid_rows].astype(float)

    if validated_data.empty:
        return input_data, "No rows with valid target values were found."

    return validated_data, None
