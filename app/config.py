import logging
import sys
from pathlib import Path
from types import FrameType
from typing import cast

from loguru import logger
from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATASET_PATH = PROJECT_ROOT / "datasets" / "test_cars.csv"


class LoggingSettings(BaseSettings):
    LOGGING_LEVEL: int = logging.INFO


class Settings:
    API_V1_STR = "/api/v1"  # Prefix in URL

    logging: LoggingSettings = LoggingSettings()

    BACKEND_CORS_ORIGINS: list[AnyHttpUrl] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://localhost:3000",
        "https://localhost:8000",
    ]

    PROJECT_NAME: str = "Car Price Prediction API"

    model_config = SettingsConfigDict(case_sensitive=True)


class InterceptHandler(logging.Handler):
    """Redirect standard logging messages to Loguru."""

    # Handle and redirect Python Logging records to Loguru
    def emit(self, record: logging.LogRecord) -> None:
        # Get the log level name from Loguru
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            # Use the numeric log level if the level name is not recognized
            level = str(record.levelno)

        # Get the current execution frame and set the initial call depth
        frame, depth = logging.currentframe(), 2

        # Skip internal frames belonging to Python's logging
        while frame and frame.f_code.co_filename == logging.__file__:
            # Move to the previous frame
            frame = cast(FrameType, frame.f_back)

            # Increase the depth to identify the original logging call
            depth += 1

        # Configure Loguru with the caller depth and exception information
        logger.opt(
            depth=depth,
            exception=record.exc_info,
        ).log(
            # Log message using the determined log level
            level,
            # Get the final formatted log message
            record.getMessage(),
        )


def setup_app_logging(config: Settings) -> None:
    """Prepare custom logging for the application."""

    # Root Logger
    logging.getLogger().handlers = [InterceptHandler()]

    # uvicorn Logger
    for logger_name in ("uvicorn.asgi", "uvicorn.access"):
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(level=config.logging.LOGGING_LEVEL)]

    logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "level": config.logging.LOGGING_LEVEL,
            }
        ]
    )


settings = Settings()
