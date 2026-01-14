from pydantic import BaseModel, Field


class ForwardRequest(BaseModel):
    secid: str = Field(default="SBER", description="Тикер")
    horizon: int = Field(ge=1, le=365, description="Горизонт прогноза в шагах")


class ForwardResponse(BaseModel):
    secid: str
    horizon: int
    target: str
    forecast: list[float]
    model: str


class HistoryItem(BaseModel):
    id: int
    created_at: str
    secid: str
    horizon: int
    payload_len: int
    processing_ms: float
    status_code: int
    error: str | None


class StatsResponse(BaseModel):
    count: int
    timings_ms: dict
    inputs: dict
