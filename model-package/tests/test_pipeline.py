from unittest.mock import patch

import numpy as np

from config.config import config
from processing.data_manager import (
    load_dataset,
    load_pipeline,
    remove_old_pipelines,
    save_pipeline,
)
from processing.pipeline import build_pipeline
from processing.validation import validate_training_data


def test_pipeline_fit_and_predict():
    """Verify that the pipeline can be fitted on training data
    and then produce valid predictions.
    """
    # Given
    input_data = load_dataset(file_name="used_cars.csv")
    validated_data, error = validate_training_data(input_data=input_data)

    assert error is None

    X = validated_data.drop(columns=[config.model_settings.target])
    y = validated_data[config.model_settings.target]

    # when
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    predictions = pipeline.predict(X.head(5))

    # then
    assert len(predictions) == 5


def test_save_and_load_pipeline(tmp_path):
    """Verify that a fitted pipeline gives identical predictions
    after being saved and reloaded.
    """
    # Given
    input_data = load_dataset(file_name="used_cars.csv")
    validated_data, error = validate_training_data(input_data=input_data)

    assert error is None

    X = validated_data.drop(columns=[config.model_settings.target])
    y = validated_data[config.model_settings.target]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    file_path = tmp_path / "test_pipeline.pkl"
    expected_predictions = pipeline.predict(X.head(5))

    # when
    save_pipeline(pipeline_to_persist=pipeline, file_path=file_path)
    loaded_pipeline = load_pipeline(file_path=file_path)
    actual_predictions = loaded_pipeline.predict(X.head(5))

    # then
    np.testing.assert_array_equal(actual_predictions, expected_predictions)


def test_remove_old_pipelines(tmp_path):
    """Verify that old pipeline files are removed and kept files remain."""

    # Given
    # define files
    kept_file = tmp_path / "keep.pkl"
    old_file = tmp_path / "old.pkl"

    # creat files
    kept_file.touch()
    old_file.touch()

    # When
    with patch("processing.data_manager.TRAINED_MODEL_DIR", tmp_path):
        remove_old_pipelines(files_to_keep=["keep.pkl"])

    # Then
    assert kept_file.exists()
    assert not old_file.exists()
