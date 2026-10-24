from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


CONFIG_FILE_PATH = PACKAGE_ROOT / "config" / "config.yml"


class VariablesConfig(BaseModel):
    mileage: list[str]
    engine: list[str]
    car_age: list[str]
    transmission: list[str]
    color: list[str]
    price: list[str]
    accident: list[str]
    fuel_type: list[str]
    clean_title: list[str]


class AppConfig(BaseModel):
    """Application-level configuration."""

    package_name: str
    dataset_name: str
    new_dataset_name: str
    incoming_dataset_dir: str
    target: str
    preprocessing_pipeline_name: str
    target_pipeline_name: str


class ModelConfig(BaseModel):
    """Configuration related to model training."""

    target: str
    features: list[str]
    numerical_vars: list[str]
    categorical_vars: list[str]
    binary_vars: list[str]
    test_size: float
    random_state: int
    n_estimators: int
    n_jobs: int
    min_frequency: int


class Config(BaseModel):
    """Master configuration object."""

    app_config: AppConfig
    model_settings: ModelConfig
    variables: VariablesConfig


def find_config_file() -> Path:
    """Locate the project configuration file."""

    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH

    raise FileNotFoundError(f"Configuration file not found at: {CONFIG_FILE_PATH}")


def fetch_config_from_yaml(
    cfg_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Load project configuration from a YAML file."""

    if cfg_path is None:
        cfg_path = find_config_file()

    with open(cfg_path, "r", encoding="utf-8") as conf_file:
        parsed_config = yaml.safe_load(conf_file)

    if not isinstance(parsed_config, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {cfg_path}")

    return parsed_config


def create_and_validate_config() -> Config:
    """Create and validate the project configuration."""

    parsed_config = fetch_config_from_yaml()

    app_config = AppConfig(
        package_name=parsed_config["package_name"],
        dataset_name=parsed_config["dataset_name"],
        new_dataset_name=parsed_config["new_dataset_name"],
        incoming_dataset_dir=parsed_config["incoming_dataset_dir"],
        target=parsed_config["target"],
        preprocessing_pipeline_name=parsed_config["preprocessing_pipeline_name"],
        target_pipeline_name=parsed_config["target_pipeline_name"],
    )

    # model_settings = model_config
    model_settings = ModelConfig(
        target=parsed_config["target"],
        features=parsed_config["features"],
        numerical_vars=parsed_config["numerical_vars"],
        categorical_vars=parsed_config["categorical_vars"],
        binary_vars=parsed_config["binary_vars"],
        test_size=parsed_config["test_size"],
        random_state=parsed_config["random_state"],
        n_estimators=parsed_config["n_estimators"],
        n_jobs=parsed_config["n_jobs"],
        min_frequency=parsed_config["min_frequency"],
    )

    return Config(
        app_config=app_config,
        model_settings=model_settings,
        variables=VariablesConfig(**parsed_config["variables"]),
    )


# Load and validate configuration
config = create_and_validate_config()


DATASET_DIR = PACKAGE_ROOT / "datasets"
DATASET_PATH = DATASET_DIR / config.app_config.dataset_name

INCOMING_DATASET_DIR = DATASET_DIR / config.app_config.incoming_dataset_dir

NEW_DATASET_PATH = INCOMING_DATASET_DIR / config.app_config.new_dataset_name


TRAINED_MODEL_DIR = (
    PACKAGE_ROOT / "car_price_model" / "trained_models"
)

PREPROCESSING_PIPELINE_PATH = (
    TRAINED_MODEL_DIR / config.app_config.preprocessing_pipeline_name
)

TARGET_PIPELINE_PATH = TRAINED_MODEL_DIR / config.app_config.target_pipeline_name


TARGET = config.model_settings.target
