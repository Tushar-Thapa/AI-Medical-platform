from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .common import BaseRequest, BaseResponse, Severity


class ExtractedCondition(BaseModel):
    """Single medical condition extracted from report text."""

    condition_name: str
    icd10_code: Optional[str] = Field(
        default=None, description="ICD-10 code if resolvable, e.g. J18.9 for pneumonia"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: Optional[Severity] = None
    negated: bool = Field(
        default=False,
        description="True if condition was negated in text (e.g. 'no fever')",
    )


class ReportAnalysisRequest(BaseRequest):
    """POST /report/analyze — raw clinical report text."""

    report_text: str = Field(..., min_length=10, description="Raw clinical report text to parse")
    patient_id: Optional[str] = None
    report_type: Optional[str] = Field(
        default=None,
        description="e.g. 'radiology', 'discharge_summary', 'lab_report'",
    )


class ReportAnalysisResponse(BaseResponse):
    """Response from POST /report/analyze. Consumed by drug_recommendation module."""

    patient_id: Optional[str] = None
    conditions: List[ExtractedCondition]
    raw_entities: List[str] = Field(
        default_factory=list,
        description="All NER entities found before filtering",
    )
    model_version: str = Field(default="bioBERT-v1")
