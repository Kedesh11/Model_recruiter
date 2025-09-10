from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Score(BaseModel):
    application_id: str
    completeness: float = Field(ge=0.0, le=100.0, default=0.0)
    fit: float = Field(ge=0.0, le=100.0, default=0.0)
    mtp: float = Field(ge=0.0, le=100.0, default=0.0)
    conformity: float = Field(ge=0.0, le=100.0, default=0.0)
    final: float = Field(ge=0.0, le=100.0, default=0.0)
    recommendation: Optional[str] = None
