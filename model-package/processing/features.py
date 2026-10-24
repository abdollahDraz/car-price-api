# Contain Custom transformers

import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MileageTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for variable in self.variables:
            if variable not in X.columns:
                raise ValueError(f"Column '{variable}' not found in X")

            values = (
                X[variable]
                .astype("string")
                .str.replace(",", "", regex=False)
                .str.replace(r"\s*mi\.?", "", regex=True, case=False)
                .str.strip()
            )

            X[variable] = pd.to_numeric(values, errors="coerce")

        return X


# in this transformer we have one edit to can be make with v(8)
class EngineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables, drop_original=True):
        self.variables = variables
        self.drop_original = drop_original

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for variable in self.variables:
            if variable not in X.columns:
                raise ValueError(f"Column '{variable}' not found in X")

            engine = X[variable].astype("string").str.strip()

            horsepower = engine.str.extract(
                r"(\d+(?:\.\d+)?)\s*HP\b",
                flags=re.IGNORECASE,
                expand=False,
            )
            engine_size = engine.str.extract(
                r"(\d+(?:\.\d+)?)\s*L\b",
                flags=re.IGNORECASE,
                expand=False,
            )
            cylinders = engine.str.extract(
                r"\b(\d+)\s*Cylinder",
                flags=re.IGNORECASE,
                expand=False,
            )
            config_cylinders = engine.str.extract(
                r"\b[VI]\s*\(\s*(\d+)\s*\)",
                flags=re.IGNORECASE,
                expand=False,
            )
            straight_cylinders = engine.str.extract(
                r"\bStraight\s+(\d+)\b",
                flags=re.IGNORECASE,
                expand=False,
            )

            cylinders = cylinders.combine_first(config_cylinders).combine_first(
                straight_cylinders
            )

            X["horsepower"] = pd.to_numeric(horsepower, errors="coerce")
            X["engine_size"] = pd.to_numeric(engine_size, errors="coerce")
            X["cylinder_count"] = pd.to_numeric(cylinders, errors="coerce")

            if self.drop_original:
                X = X.drop(columns=[variable])

        return X


class CarAgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        variables,
        output_variable="car_age",
        reference_year=2026,
    ):
        self.variables = variables
        self.output_variable = output_variable
        self.reference_year = reference_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for variable in self.variables:
            if variable not in X.columns:
                raise ValueError(f"Column '{variable}' not found in X")

            output_name = (
                self.output_variable if len(self.variables) == 1 else f"{variable}_age"
            )

            model_year = pd.to_numeric(X[variable], errors="coerce")
            X[output_name] = self.reference_year - model_year

        return X


class TransmissionMapper(BaseEstimator, TransformerMixin):
    def __init__(self, variables, mapper):
        self.variables = variables
        self.mapper = mapper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for variable in self.variables:
            if variable not in X.columns:
                raise ValueError(f"Column '{variable}' not found in X")

            X[f"{variable}_type"] = (
                X[variable]
                .apply(self.mapper)
                .astype(object)
                .where(lambda s: s.notna(), float("nan"))
            )

        return X


class ColorMapper(BaseEstimator, TransformerMixin):
    def __init__(self, variables, mapper, drop_original=True):
        self.variables = variables
        self.mapper = mapper
        self.drop_original = drop_original

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for variable in self.variables:
            if variable not in X.columns:
                raise ValueError(f"Column '{variable}' not found in X")

            output_variable = f"{variable}_group"

            X[output_variable] = (
                X[variable]
                .apply(self.mapper)
                .astype(object)
                .where(lambda s: s.notna(), float("nan"))
            )

            if self.drop_original:
                X = X.drop(columns=[variable])

        return X


# for training new data
class PriceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for variable in self.variables:
            if variable not in X.columns:
                raise ValueError(f"Column '{variable}' not found in X")

            values = (
                X[variable]
                .astype("string")
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )

            X[variable] = pd.to_numeric(values, errors="coerce")

        return X


class AccidentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables, mapper):
        self.variables = variables
        self.mapper = mapper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for variable in self.variables:
            if variable not in X.columns:
                raise ValueError(f"Column '{variable}' not found in X")

            X[variable] = pd.to_numeric(
                X[variable].apply(self.mapper),
                errors="coerce",
            ).astype("float64")

        return X


class FuelTypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for variable in self.variables:
            if variable not in X.columns:
                raise ValueError(f"Column '{variable}' not found in X")

            values = X[variable].astype("string").str.strip()

            invalid_values = values.str.lower().isin(
                {"", "-", "–", "not supported", "nan", "none"}
            )

            values = values.mask(invalid_values, pd.NA)
            values = values.astype(object)
            values = values.where(pd.notna(values), float("nan"))

            X[variable] = values

        return X


class CleanTitleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables, mapper):
        self.variables = variables
        self.mapper = mapper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for variable in self.variables:
            if variable not in X.columns:
                raise ValueError(f"Column '{variable}' not found in X")

            X[variable] = pd.to_numeric(
                X[variable].apply(self.mapper),
                errors="coerce",
            ).astype("float64")

        return X


# transmission mapper
def map_transmission(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).strip().lower()

    if value in {
        "",
        "-",
        "–",
        "nan",
        "none",
        "f",
        "2",
        "scheduled for or in production",
    }:
        return pd.NA

    if "single-speed" in value or "single speed" in value:
        return "Single-Speed"

    if "cvt" in value or "variable" in value:
        return "CVT"

    if (
        "dct" in value
        or "dual-clutch" in value
        or "dual clutch" in value
        or "pdk" in value
        or "s tronic" in value
    ):
        return "DCT"

    if "manual" in value or "m/t" in value or re.search(r"\bmt\b", value):
        return "Manual"

    if (
        "automatic" in value
        or "a/t" in value
        or "at/mt" in value
        or re.search(r"\bat\b", value)
        or re.search(r"\bauto\b", value)
        or "cmdshft" in value
        or "overdrive switch" in value
    ):
        return "Automatic"

    return pd.NA


# Color Rules
COLOR_RULES = [
    (
        "Black",
        [
            "black",
            "blk",
            "ebony",
            "obsidian",
            "onyx",
            "nero",
            "caviar",
            "raven",
            "beluga",
            "noir",
        ],
    ),
    (
        "White",
        [
            "white",
            "ivory",
            "bianco",
            "glacier",
            "pearl white",
            "arctic white",
            "snow white",
        ],
    ),
    (
        "Gray",
        [
            "gray",
            "grey",
            "silver",
            "grigio",
            "graphite",
            "slate",
            "ash",
            "steel",
            "granite",
            "platinum",
            "titanium",
            "quartz",
            "charcoal",
            "pewter",
        ],
    ),
    (
        "Red",
        [
            "red",
            "rosso",
            "garnet",
            "ruby",
            "maroon",
            "crimson",
            "scarlet",
            "sangria",
            "cayenne",
            "cranberry",
            "pimento",
        ],
    ),
    ("Blue", ["blue", "blu", "navy", "sapphire", "cobalt", "indigo"]),
    ("Green", ["green", "verde", "moss", "emerald", "cypress", "lizard", "python"]),
    (
        "Brown",
        [
            "brown",
            "beige",
            "tan",
            "bronze",
            "camel",
            "sand",
            "walnut",
            "chestnut",
            "mocha",
            "cappuccino",
            "saddle",
            "caramel",
            "espresso",
            "cocoa",
        ],
    ),
    ("Yellow", ["yellow", "gold", "hellayella"]),
    ("Orange", ["orange", "arancio", "mango", "amber"]),
    ("Purple", ["purple", "plum", "orchid", "violet"]),
    ("Pink", ["pink"]),
]


# Color Mapper
def map_color(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).strip().lower()

    if value in {"", "-", "–", "nan", "none"}:
        return pd.NA

    for color, keywords in COLOR_RULES:
        if any(keyword in value for keyword in keywords):  # any catch name of Color
            return color

    return pd.NA


# Accident Mapper
def map_accident(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).strip().lower()

    mapping = {"none reported": 0, "at least 1 accident or damage reported": 1}

    return mapping.get(value, pd.NA)


# Titel MApper
def map_title(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).strip().lower()

    mapping = {"yes": 1, "no": 0}

    return mapping.get(value, pd.NA)
