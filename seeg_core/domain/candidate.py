from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


class Candidate(BaseModel):
    id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    gender: Optional[str] = Field(default=None, description="'M'/'F'/None")
    age: Optional[int] = None


class ApplicationDocument(BaseModel):
    id: str
    application_id: str
    doc_type: str
    text: Optional[str] = None
    url: Optional[str] = None


class Application(BaseModel):
    id: str
    candidate_id: str
    job_offer_id: str
    documents: List[ApplicationDocument] = []
