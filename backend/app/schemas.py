"""Pydantic response/request models for the API."""

from typing import Optional

from pydantic import BaseModel


class SegmentResponse(BaseModel):
    patient_id: str
    visit_date: str
    area_mm2: float
    overlay_url: str


class VisitRow(BaseModel):
    id: Optional[str] = None
    patient_id: str
    visit_date: str
    area_mm2: float
    overlay_url: str
    created_at: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
