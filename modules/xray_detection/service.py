from __future__ import annotations

import time

from shared.schemas.xray import XRayRequest, XRayResponse, XRayPrediction, XRayLabel
from shared.schemas.common import Severity
from shared.utils.image_utils import decode_base64_image
from shared.utils.exceptions import InvalidImageError, InferenceFailed
from modules.xray_detection.model import load_model, predict


class XRayService:
    def __init__(self):
        self._model = load_model()
    
    def run(self,request: XRayRequest) -> XRayResponse:
        start = time.time()

        try:
            image =decode_base64_image(request.image_base64)
        except ValueError as e:
            raise InvalidImageError(str(e))

        try:
            label_str, confidence = predict(self._model,image)
        except Exception as e:
            raise InferenceFailed("xray_detection", str(e))

        label = XRayLabel(label_str)
        severity = Severity.HIGH if label == XRayLabel.PNEUMONIA else Serverity.LOW
        
        return XRayResponse(
            request_id = request.request_id,
            processing_time_ms=(time.time()-start)*1000,
            patient_id= request.patient_id,
            prediction = XRayPrediction(
                label=label,
                confidence =confidence,
                severity=severity,
            ),
        )