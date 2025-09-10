from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class JobOffer(BaseModel):
    id: str
    title: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None


class JobText(BaseModel):
    job_offer_id: str
    text: str = ""
