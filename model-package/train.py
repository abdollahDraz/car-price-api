# training model by new data

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config.config import (
    PREPROCESSING_PIPELINE_PATH,
    TARGET_PIPELINE_PATH,
    config,
)
from processing.data_manager import (
    combine_datasets,
    load_pipeline,
    load_preprocessing_pipeline,
    remove_old_pipelines,
    save_pipeline,
)
from processing.pipeline import build_pipeline
from processing.validation import validate_training_data


def run_training() -> Pipeline:
    """Train and return the best-performing pipeline."""

    # Load and combine the available datasets
    data = combine_datasets()

    # Validate the training dataset
    data, error = validate_training_data(input_data=data)

    if error is not None:
        raise ValueError(f"Training data validation failed: {error}")

    # Separate predictors and target
    X = data.drop(columns=[config.model_settings.target])
    y = data[config.model_settings.target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.model_settings.test_size,
        random_state=config.model_settings.random_state,
    )

    # Load the existing baseline
    baseline_pipeline = load_preprocessing_pipeline()

    # Evaluate the baseline
    baseline_predictions = baseline_pipeline.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)

    # Create an independent candidate and retrain it
    candidate_pipeline = build_pipeline()
    candidate_pipeline.fit(X_train, y_train)

    # Evaluate the candidate
    candidate_predictions = candidate_pipeline.predict(X_test)
    candidate_mae = mean_absolute_error(y_test, candidate_predictions)

    print("Baseline MAE:", round(baseline_mae, 2))
    print("Candidate MAE:", round(candidate_mae, 2))

    # Select the best-performing pipeline
    if candidate_mae < baseline_mae:
        print("Selected model: Candidate")
        return candidate_pipeline

    print("Selected model: Baseline")
    return baseline_pipeline


def train_and_save_best_model() -> None:
    """Train and save the best-performing pipeline."""

    best_pipeline = run_training()

    temporary_pipeline_path = PREPROCESSING_PIPELINE_PATH.with_name(
        f"{PREPROCESSING_PIPELINE_PATH.stem}_temporary"
        f"{PREPROCESSING_PIPELINE_PATH.suffix}"
    )

    save_pipeline(
        pipeline_to_persist=best_pipeline,
        file_path=temporary_pipeline_path,
    )

    # Verify that the temporary pipeline can be loaded
    load_pipeline(file_path=temporary_pipeline_path)

    # Replace the production pipeline only after saving succeeds
    temporary_pipeline_path.replace(PREPROCESSING_PIPELINE_PATH)

    # Keep both required pipelines and remove obsolete versions
    remove_old_pipelines(
        files_to_keep=[
            PREPROCESSING_PIPELINE_PATH.name,
            TARGET_PIPELINE_PATH.name,
        ]
    )

    print("Best pipeline saved successfully.")


if __name__ == "__main__":
    train_and_save_best_model()
