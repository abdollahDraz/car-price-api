import json

import numpy as np
import pandas as pd
from car_price_model import __version__ as model_version
from car_price_model.predict import make_prediction
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger

from app import __version__ as api_version
from app.config import Settings
from app.schemas.health import HealthResponse
from app.schemas.predict import MultipleCarDataInputsRequest, PredictionResults

api_router = APIRouter()


@api_router.get("/health", response_model=HealthResponse, status_code=200)
def health() -> HealthResponse:
    """Check API health and report model version."""
    return HealthResponse(
        name=Settings.PROJECT_NAME,
        api_version=api_version,
        model_version=model_version,
        description="API for predicting used car prices",
    )


@api_router.post("/predict", response_model=PredictionResults, status_code=200)
async def predict(input_data: MultipleCarDataInputsRequest) -> PredictionResults:
    """Predict used car prices."""

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))  # create dataframe
    logger.info(  # log -> info for prediction
        "Making prediction for {}car(s)", len(input_df)
    )

    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(  # log -> warning for prediction have errors
            "Prediction validation error: {}", results["errors"]
        )
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))
    logger.info(  # log -> info for results of prediction
        "Prediction results: {}", results["predictions"]
    )

    return results
