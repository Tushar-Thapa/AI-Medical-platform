from __future__ import annotations

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding="utf-8",
        case_sensitive = False, 
    )

    app_title: str = "AI Medical Platform"
    app_version: str = "0.1.0"
    debug: bool = False

    enable_xray: bool = True
    enable_report: bool = True
    enable_drug: bool = True
    enable_tumor: bool =True

    xray_model_path: Optional[str] = "data/models/resnet18_xray.pth"
    report_model_name: str = "allenai/scibert_scivocab_uncased"
    tumor_model_path: Optional[str] = "data/models/unet_tumor.pth"
    max_image_size_mb: float = 10.0
    inference_timeout_seconds: float = 30.0

setting = Settings()