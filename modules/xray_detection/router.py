from __future__ import annotations

from fastapi import APIRouter
from shared.schemas.xray import XRayRequest, XRayResponse
from shared.schemas.common import HealthStatus
from modules.xray_detection.service import XRayService

router = APIRouter()
_service = XRayService()

@router.get("/health",response_model=HealthStatus)
async def health():
    return HealthStatus(status= "ok", module="xray_detection")


@router.post("/predict", response_model = XRayResponse)
async def predict(request: XRayRequest) -> XRayResponse:
    return _service.run(request)