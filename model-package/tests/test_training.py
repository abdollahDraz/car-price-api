import shutil

import pandas as pd
from sklearn.pipeline import Pipeline

import processing.data_manager as data_manager
import train
from config.config import config
from processing.data_manager import combine_datasets
from processing.validation import validate_training_data
from train import run_training


def test_run_training_returns_fitted_pipeline():
    """Test training with the project's real dataset and saved baseline"""
    # Given

    # loading data
    data = combine_datasets()

    # validated the input data for training
    data, error = validate_training_data(input_data=data)

    assert error is None
    assert not data.empty

    # select best model
    selected_pipeline = run_training()  # Pipeline

    # prepare X
    X = data.drop(columns=[config.model_settings.target]).head(5)

    # when
    predictions = selected_pipeline.predict(X)

    # then
    assert isinstance(selected_pipeline, Pipeline)
    assert len(predictions) == len(X)
    assert pd.notna(predictions).all()


def test_train_and_save_best_model(
    tmp_path, monkeypatch
):  # tmp_path & monkeypatch  -> fixtures
    """Test saving the selected model pipeline"""
    # Given
    # creat folder in temporary folder
    temporary_model_dir = tmp_path / "trained_models"
    temporary_model_dir.mkdir()

    # creat another path for Preprocessing path  and target path
    preprocessing_path = temporary_model_dir / train.PREPROCESSING_PIPELINE_PATH.name

    target_path = temporary_model_dir / train.TARGET_PIPELINE_PATH.name

    # copy target pipeline file for temporary path
    shutil.copy2(train.TARGET_PIPELINE_PATH, target_path)

    # change value of  PREPROCESSING_PIPELINE_PATH Temporary
    monkeypatch.setattr(train, "PREPROCESSING_PIPELINE_PATH", preprocessing_path)
    monkeypatch.setattr(
        train,
        "TARGET_PIPELINE_PATH",
        target_path,
    )

    # change value of  TRAINED_MODEL_DIR temporary
    monkeypatch.setattr(data_manager, "TRAINED_MODEL_DIR", temporary_model_dir)
    # load pipeline
    existing_pipeline = train.load_preprocessing_pipeline()
    # changing the behavior of this func to return existing_pipeline temporary
    monkeypatch.setattr(
        train, "run_training", lambda: existing_pipeline  # func  # func
    )

    # creat file in temporary folder and will be remove this file
    obsolete_path = temporary_model_dir / "obsolete.pkl"  # define path of file
    obsolete_path.touch()  # creat file

    # when
    train.train_and_save_best_model()

    # then
    assert preprocessing_path.exists()
    assert target_path.exists()
    assert not obsolete_path.exists()

    temporary_path = preprocessing_path.with_name(
        f"{preprocessing_path.stem}_temporary" f"{preprocessing_path.suffix}"
    )

    assert not temporary_path.exists()

    # check if the new pipeline is working
    saved_pipeline = train.load_pipeline(file_path=preprocessing_path)

    assert isinstance(saved_pipeline, Pipeline)

    data = combine_datasets()

    data, error = validate_training_data(input_data=data)

    assert error is None

    X = data.drop(columns=[config.model_settings.target]).head(5)

    predictions = saved_pipeline.predict(X)

    assert len(predictions) == len(X)
    assert pd.notna(predictions).all()
