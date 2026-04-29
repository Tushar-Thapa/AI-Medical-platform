from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Severity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class Modality(str, Enum):
    XRAY = "xray"
    CT = "ct"
    MRI = "mri"
    REPORT = "report"


class HealthStatus(BaseModel):
    status: str = Field(default="ok")
    module: str
    version: str = Field(default="0.1.0")


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


class BaseRequest(BaseModel):
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for tracing this request end-to-end",
    )


class BaseResponse(BaseModel):
    request_id: str
    processing_time_ms: Optional[float] = None
    
