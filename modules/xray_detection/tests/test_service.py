from __future__ import annotations

import base64
import pytest
from unittest.mock import MagicMock, patch
from shared.schemas.xray import XRayRequest, XRayLabel
from module.xray_detection.service import XRayService


def make_dummy_image_b64() -> str:
    from PIL import Image
    import io
    img = Image.new("RGB", (224,224), color = (128,128,128))
    buffer = io.BytesIO()
    img.save(buffer,format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@patch("module.xray_detection.service.load_model")
def test_predict_pneumonia(mock_load):
    mock_model = MagicMock()
    mock_load.return_value = mock_model

    with patch("modules.xray_detection.service.predict", return_value=("PNEUMONIA",0.91)):
        service = XRayService()
        request = XRayRequest(
            image_base64=make_dummy_image_b64(),
            patient_id = "patient-001",
        )
        response = service.run(request)

    assert response.prediction.label == XRayLabel.PNEUMONIA
    assert response.prediction.confidence == 0.91
    assert response.pateint == "patient-001"