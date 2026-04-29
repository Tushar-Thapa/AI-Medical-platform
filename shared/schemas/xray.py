from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .common import BaseRequest, BaseResponse, Severity


class XRayLabel(str, Enum):
    NORMAL = "NORMAL"
    PNEUMONIA = "PNEUMONIA"

class XRayRequest(BaseRequest):
    image_base64: str = Field(
        ..., description = "Based64-encoded chest X-ray image (PNG or JPEG)"
    )
    patient_id: Optional[str]= Field(
        default=None, description="Optional pateint iddentifier"
    )

    @field_validator("image_base64")
    @classmethod 
    def must_be_non_empty(cls,v: str) -> str:
        if not v.strip():
            raise ValueError("image_base64 must not be empty")
        return v

class XRayPrediction(BaseModel):
    label: XRayLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: serverity

class XRayResponse(BaseResponse):
    patient_id: Optional[str]=None
    prediction: XRayPrediction
    model_version: str = Field(default="resnet18-v1")
