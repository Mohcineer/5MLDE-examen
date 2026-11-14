"""Schémas Pydantic de l'API : les bornes reprennent celles des contrôles qualité des lots."""
from pydantic import BaseModel, ConfigDict, Field

MAX_BATCH_ITEMS = 100


class BikeFeatures(BaseModel):
    """Conditions d'une heure donnée ; aucune colonne qui fuit la cible (casual, registered)."""
    model_config = ConfigDict(extra="forbid")

    season: int = Field(ge=1, le=4, description="1 hiver, 2 printemps, 3 été, 4 automne")
    yr: int = Field(ge=0, le=1, description="0 = 2011, 1 = 2012")
    mnth: int = Field(ge=1, le=12)
    hr: int = Field(ge=0, le=23)
    holiday: int = Field(ge=0, le=1)
    weekday: int = Field(ge=0, le=6)
    workingday: int = Field(ge=0, le=1)
    weathersit: int = Field(ge=1, le=4)
    temp: float = Field(ge=0.0, le=1.0, description="température normalisée")
    atemp: float = Field(ge=0.0, le=1.0, description="température ressentie normalisée")
    hum: float = Field(ge=0.0, le=1.0)
    windspeed: float = Field(ge=0.0, le=1.0)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[BikeFeatures] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    model_version: str


class Prediction(BaseModel):
    prediction_id: str
    predicted_cnt: float = Field(description="locations prévues sur l'heure (>= 0)")


class PredictResponse(ModelInfo, Prediction):
    request_id: str


class BatchResponse(ModelInfo):
    request_id: str
    predictions: list[Prediction]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: str = Field(min_length=1, max_length=64)
    observed_cnt: float = Field(ge=0, le=10000, description="valeur réelle observée")


class FeedbackResponse(BaseModel):
    feedback_id: str
    prediction_id: str
    model_version: str
    absolute_error: float


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None
