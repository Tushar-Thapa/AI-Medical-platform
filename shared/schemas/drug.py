from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .common import BaseRequest, BaseResponse, Severity
from .report import ExtractedCondition
from .xray import XRayLabel


class DrugRecommendation(BaseModel):
    drug_name: str
    generic_name: Optional[str] = None
    dosage: str
    route: str = Field(description="e.g. 'oral', 'IV', 'inhaled'")
    duration_days: Optional[int] = None
    contraindications: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


class DrugInteraction(BaseModel):
    drug_a: str
    drug_b: str
    interaction_level: Severity
    description: str


class DrugRecommendationRequest(BaseRequest):
    """
    POST /drug/recommend — combines xray diagnosis + NLP-extracted conditions.
    Either field alone is valid; both together yields richer recommendations.
    """

    patient_id: Optional[str] = None
    xray_label: Optional[XRayLabel] = Field(
        default=None, description="Classification result from xray_detection module"
    )
    xray_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    conditions: List[ExtractedCondition] = Field(
        default_factory=list,
        description="Conditions extracted by report_analyzer module",
    )
    patient_age: Optional[int] = Field(default=None, ge=0, le=150)
    patient_weight_kg: Optional[float] = Field(default=None, gt=0.0)
    known_allergies: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)


class DrugRecommendationResponse(BaseResponse):
    """Response from POST /drug/recommend."""

    patient_id: Optional[str] = None
    recommendations: List[DrugRecommendation]
    interactions_detected: List[DrugInteraction] = Field(default_factory=list)
    disclaimer: str = Field(
        default=(
            "This output is for clinical decision support only. "
            "Final prescribing decisions must be made by a licensed physician."
        )
    )
    model_version: str = Field(default="drug-cds-v1")
