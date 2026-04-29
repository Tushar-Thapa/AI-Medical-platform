from __future__ import annotations

import torch 
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image 

from shared.utils.config import settings
from shared.utils.exceptions import ModelNotLoadedError, InferenceFailed

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
])

def load_model() -> nn.Module:
    model = models.resnet18(weights = None)
    model.fc = nn.Linear(model.fc.in_features,2)

    try:
        state_dict = torch.load(settings.xray_model_path, map_location="cpu")
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        raise ModelNotLoadedError("xray_detection")

    model.eval()
    return model

def predict(model: nn.Module, image: Image.Image) -> tuple[str,float]:
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output,dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
    
    labels = ["NORMAL","PNEUMONIA"]
    return labels[predicted.item()], confidence.item()