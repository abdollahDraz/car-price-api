from processing.data_manager import load_dataset
from processing.validation import validate_training_data, validation_inputs


def test_validate_training_data_with_valid_data():
    """Verify that validate_training_data accepts a valid dataset
    without errors.
    """
    # Given
    input_data = load_dataset(file_name="used_cars.csv")

    # when
    validated_data, error = validate_training_data(input_data=input_data)

    # then
    assert error is None
    assert "price" in validated_data.columns
    assert len(validated_data) == len(input_data)


def test_validate_training_data_without_target():
    """Verify that validate_training_data raises an error
    when the target column is missing.
    """

    # Given
    input_data = load_dataset(file_name="used_cars.csv")
    input_data = input_data.drop(columns=["price"])

    # when
    validated_data, error = validate_training_data(input_data=input_data)

    # then
    assert error is not None


def test_validate_training_data_with_missing_price():
    """Verify that validate_training_data handles missing
    values in the price column.
    """

    # Given
    input_data = load_dataset(file_name="used_cars.csv")
    input_data.loc[0, "price"] = None

    # when
    validated_data, error = validate_training_data(input_data=input_data)

    # then
    assert error is None
    assert len(validated_data) == len(input_data) - 1


def test_validation_inputs_without_required_feature():
    """Verify that input validation raises an error
    when a required feature is missing.
    """

    # Given
    input_data = load_dataset(file_name="used_cars.csv")
    input_data = input_data.drop(columns=["brand"])

    # When
    validated_data, error = validation_inputs(input_data=input_data)

    # Then
    assert error is not None


def test_validation_inputs_with_invalid_year():
    """Verify that input validation raises an error when
    model_year is outside the valid range.
    """

    # Given
    input_data = load_dataset(file_name="used_cars.csv")
    input_data["model_year"] = input_data["model_year"].astype(object)
    input_data.loc[input_data.index[0], "model_year"] = "not-a-year"

    # When
    validated_data, error = validation_inputs(input_data=input_data)

    # Then
    assert error is not None
