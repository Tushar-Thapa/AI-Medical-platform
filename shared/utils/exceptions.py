from __future__ import annotations

from fastapi import HTTPException, status


class ModelNotLoadedError(HTTPException):
    def __init__(self,module: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail= f"Model for module '{module}' is not loaded. "
                f"Check model path in .env and run scripts/download_models.py",
        )

class InvalidImageError(HTTPException):
    def __init__(self,detail:str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid image input: {detail}",
        )

class InferenceFailed(HTTPException):
    def __init__(self, module:str, reason:str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed in module '{module}': {reason}",
        )