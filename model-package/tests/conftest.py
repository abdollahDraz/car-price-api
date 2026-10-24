import pandas as pd
import pytest


@pytest.fixture
def sample_input_data():

    return pd.DataFrame(
        [
            {
                "brand": "Ford",
                "model": "Utility Police Interceptor Base",
                "model_year": 2013,
                "milage": "51,000 mi.",
                "fuel_type": "E85 Flex Fuel",
                "engine": "300.0HP 3.7L V6 Cylinder Engine Flex Fuel Capability",
                "transmission": "6-Speed A/T",
                "ext_col": "Black",
                "int_col": "Black",
                "accident": "At least 1 accident or damage reported",
                "clean_title": "Yes",
            }
        ]
    )
