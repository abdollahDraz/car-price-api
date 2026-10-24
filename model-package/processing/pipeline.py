from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config.config import config
from processing import features as pp

MILEAGE_VARS = config.variables.mileage
ENGINE_VARS = config.variables.engine
CAR_AGE_VARS = config.variables.car_age
TRANSMISSION_VARS = config.variables.transmission
COLOR_VARS = config.variables.color
ACCIDENT_VARS = config.variables.accident
FUEL_TYPE_VARS = config.variables.fuel_type
CLEAN_TITLE_VARS = config.variables.clean_title

NUMERICAL_VARS = config.model_settings.numerical_vars
CATEGORICAL_VARS = config.model_settings.categorical_vars
BINARY_VARS_WITH_NA = config.model_settings.binary_vars


def build_pipeline() -> Pipeline:
    """Build the car price prediction pipeline"""

    return Pipeline(
        steps=[
            # Feature Engineering
            (
                "mileage",
                pp.MileageTransformer(variables=MILEAGE_VARS),
            ),
            (
                "engine",
                pp.EngineTransformer(
                    variables=ENGINE_VARS,
                    drop_original=True,
                ),
            ),
            (
                "car_age",
                pp.CarAgeTransformer(
                    variables=CAR_AGE_VARS,
                    output_variable="car_age",
                    reference_year=2026,
                ),
            ),
            (
                "transmission",
                pp.TransmissionMapper(
                    variables=TRANSMISSION_VARS,
                    mapper=pp.map_transmission,
                ),
            ),
            (
                "colors",
                pp.ColorMapper(
                    variables=COLOR_VARS,
                    mapper=pp.map_color,
                    drop_original=True,
                ),
            ),
            (
                "fuel_type",
                pp.FuelTypeTransformer(variables=FUEL_TYPE_VARS),
            ),
            (
                "accident",
                pp.AccidentTransformer(
                    variables=ACCIDENT_VARS,
                    mapper=pp.map_accident,
                ),
            ),
            (
                "clean_title",
                pp.CleanTitleTransformer(
                    variables=CLEAN_TITLE_VARS,
                    mapper=pp.map_title,
                ),
            ),
            # Imputation + Encoding
            (
                "preprocessing",
                ColumnTransformer(
                    transformers=[
                        (
                            "numerical",
                            SimpleImputer(strategy="median"),
                            NUMERICAL_VARS,
                        ),
                        (
                            "categorical",
                            Pipeline(
                                steps=[
                                    (
                                        "imputer",
                                        SimpleImputer(strategy="most_frequent"),
                                    ),
                                    (
                                        "encoder",
                                        OneHotEncoder(
                                            handle_unknown="ignore",
                                            min_frequency=(
                                                config.model_settings.min_frequency
                                            ),
                                        ),
                                    ),
                                ]
                            ),
                            CATEGORICAL_VARS,
                        ),
                        (
                            "binary",
                            SimpleImputer(strategy="most_frequent"),
                            BINARY_VARS_WITH_NA,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            # Model
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=config.model_settings.n_estimators,
                    random_state=config.model_settings.random_state,
                    n_jobs=config.model_settings.n_jobs,
                ),
            ),
        ]
    )
