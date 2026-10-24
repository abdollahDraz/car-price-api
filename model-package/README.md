# Car Price Model

A Python package for predicting used car prices using a trained scikit-learn pipeline.

## Features

- Validates input data before prediction.
- Applies the fitted preprocessing pipeline.
- Predicts used car prices.
- Returns predictions, package version, and validation errors.

## Installation

```bash
pip install car-price-model
```

## Usage

```python
import pandas as pd
from car_price_model.predict import make_prediction

input_data = pd.DataFrame([
    {
        "brand": "Toyota",
        "model": "Camry",
        "model_year": 2020,
        "milage": "50,000 mi.",
        "fuel_type": "Gasoline",
        "engine": "2.5L I4 203hp",
        "transmission": "Automatic",
        "ext_col": "White",
        "int_col": "Black",
        "accident": "None reported",
        "clean_title": "Yes",
    }
])

result = make_prediction(input_data=input_data)
print(result)
```

## Output

The prediction function returns a dictionary containing predictions, the package version, and any errors.
