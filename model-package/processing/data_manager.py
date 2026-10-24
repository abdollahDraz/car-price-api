
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from config.config import (
    DATASET_DIR,
    NEW_DATASET_PATH,
    PREPROCESSING_PIPELINE_PATH,
    TARGET_PIPELINE_PATH,
    TRAINED_MODEL_DIR,
)


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """Load the original dataset."""

    file_path = DATASET_DIR / file_name
    dataframe = pd.read_csv(file_path)

    return dataframe


def load_new_dataset() -> pd.DataFrame:
    """Load the incoming dataset containing new car records."""

    dataframe = pd.read_csv(NEW_DATASET_PATH)

    return dataframe


def load_pipeline(*, file_path: Path) -> Pipeline:
    """Load a persisted scikit-learn pipeline."""

    trained_pipeline = joblib.load(filename=file_path)

    return trained_pipeline


def save_pipeline(
    *,
    pipeline_to_persist: Pipeline,
    file_path: Path,
) -> None:
    """Persist a selected scikit-learn pipeline."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline_to_persist, filename=file_path)


def load_preprocessing_pipeline() -> Pipeline:
    """Load the fitted preprocessing and prediction pipeline."""

    return load_pipeline(file_path=PREPROCESSING_PIPELINE_PATH)


def load_target_pipeline() -> Pipeline:
    """Load the target transformation pipeline."""

    return load_pipeline(file_path=TARGET_PIPELINE_PATH)


def combine_datasets() -> pd.DataFrame:
    """Combine the original dataset with incoming car records."""

    original_data = load_dataset(file_name="used_cars.csv")
    new_data = load_new_dataset()

    if list(original_data.columns) != list(new_data.columns):
        raise ValueError("The datasets have different columns.")

    combined_data = pd.concat(
        [original_data, new_data],
        ignore_index=True,
    )

    return combined_data


def remove_old_pipelines(*, files_to_keep: list[str]) -> None:
    """Remove outdated pipeline files from the trained models directory."""

    files_to_keep_set = set(files_to_keep)

    for file_path in TRAINED_MODEL_DIR.iterdir():
        if (
            file_path.is_file()
            and file_path.suffix == ".pkl"
            and file_path.name not in files_to_keep_set
        ):
            file_path.unlink()
