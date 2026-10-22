from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response schema for the health endpoint."""

    name: str
    api_version: str
    model_version: str
    description: str
