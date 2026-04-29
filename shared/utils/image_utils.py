from __future__ import annotations

import base64 
import io 
from typing import Tuple 

from PIL import Image 

def decode_base64_image(image_b64: str) -> Image.Image:
    try:
        raw_bytes = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to decode base64 image: {exc}") from exc

def encode_image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def resize_for_model(
    image: Image.Image,
    target_size: Tuple[int,int] = (224,224),
) -> Image.Image:
    return image.resize(target_size, Image.LANCZOS)
