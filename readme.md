# Plan: AI Medical Platform — Multi-Model Monorepo

## Context
Building a 4-module AI medical platform in a shared git repo (2 devs). Modules must communicate via clean API contracts defined as Pydantic schemas. Both devs need common setup before writing any module code.

**Modules:**
| Module | Owner | Tech | Endpoint prefix |
|---|---|---|---|
| xray_detection | User A | ResNet18, PyTorch | `/xray` |
| report_analyzer | User B | BioBERT/spaCy NLP | `/report` |
| drug_recommendation | User A | Rule engine + sklearn | `/drug` |
| tumor_segmentation | User A | U-Net (MONAI) | `/tumor` |

---

## Step 0: What BOTH Devs Install Before Anything Else

```bash
# ONE TIME SETUP — both devs run this
git clone <repo-url>
cd ai-doctor
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements/base.txt
cp .env.example .env       # then edit paths
```

**`requirements/base.txt`** (both devs, always):
```
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9
pydantic==2.7.1
pydantic-settings==2.2.1
httpx==0.27.0
python-dotenv==1.0.1
structlog==24.1.0
pytest==8.2.0
pytest-asyncio==0.23.6
black==24.4.2
isort==5.13.2
flake8==7.0.0
```

**User A additionally installs:**
```bash
pip install -r requirements/xray_detection.txt      # torch, torchvision, Pillow, numpy, opencv
pip install -r requirements/drug_recommendation.txt # pandas, scikit-learn, requests
pip install -r requirements/tumor_segmentation.txt  # torch, monai, nibabel, scipy
```

**User B additionally installs:**
```bash
pip install -r requirements/report_analyzer.txt     # transformers, spacy, en_core_sci_md, torch
```

---

## Folder Structure

```
ai-doctor/
├── .gitignore
├── .env.example
├── README.md
├── requirements/
│   ├── base.txt                  ← BOTH devs install this first
│   ├── xray_detection.txt
│   ├── report_analyzer.txt
│   ├── drug_recommendation.txt
│   └── tumor_segmentation.txt
├── shared/
│   ├── schemas/
│   │   ├── __init__.py           ← exports everything
│   │   ├── common.py             ← BaseRequest, BaseResponse, Severity, ErrorResponse
│   │   ├── xray.py               ← XRayRequest, XRayResponse, XRayLabel
│   │   ├── report.py             ← ReportAnalysisRequest/Response, ExtractedCondition
│   │   ├── drug.py               ← DrugRecommendationRequest/Response (imports from xray+report)
│   │   └── tumor.py              ← TumorSegmentationRequest/Response, SegmentationMask
│   └── utils/
│       ├── config.py             ← Settings (pydantic-settings, reads .env)
│       ├── image_utils.py        ← decode_base64_image, encode_image_to_base64, resize_for_model
│       ├── logger.py             ← structlog setup
│       └── exceptions.py        ← ModelNotLoadedError, InvalidImageError, InferenceFailed
├── modules/
│   ├── xray_detection/
│   │   ├── router.py             ← GET /health, POST /predict
│   │   ├── model.py              ← ResNet18 load + inference
│   │   ├── service.py            ← XRayService.predict()
│   │   └── tests/
│   ├── report_analyzer/
│   │   ├── router.py             ← GET /health, POST /analyze
│   │   ├── model.py              ← NLP pipeline
│   │   ├── service.py
│   │   └── tests/
│   ├── drug_recommendation/
│   │   ├── router.py             ← GET /health, POST /recommend
│   │   ├── model.py              ← decision engine
│   │   ├── service.py
│   │   └── tests/
│   └── tumor_segmentation/
│       ├── router.py             ← GET /health, POST /segment
│       ├── model.py              ← U-Net / MONAI inference
│       ├── service.py
│       └── tests/
├── gateway/
│   ├── main.py                   ← FastAPI app, lifespan, middleware, error handlers
│   └── router_registry.py        ← mounts all 4 module routers (respects enable_* flags)
├── scripts/
│   ├── download_models.py
│   └── setup_dev.sh
└── data/                         ← GITIGNORED
    ├── models/        (.pth files go here)
    └── datasets/      (never committed)
```

---

## API Contracts (Pydantic Schemas in `shared/schemas/`)

**Creation order matters — build common.py first:**

### `common.py` — base types
```python
class Severity(str, Enum): LOW | MODERATE | HIGH | CRITICAL
class BaseRequest(BaseModel): request_id: str (uuid auto-generated)
class BaseResponse(BaseModel): request_id: str, processing_time_ms: float
class ErrorResponse(BaseModel): error: str, detail: str, request_id: str
```

### `xray.py`
```python
class XRayRequest(BaseRequest):
    image_base64: str          # base64-encoded PNG/JPEG
    patient_id: Optional[str]

class XRayResponse(BaseResponse):
    patient_id: Optional[str]
    prediction:
        label: "NORMAL" | "PNEUMONIA"
        confidence: float 0-1
        severity: Severity
    model_version: str
```

### `report.py`
```python
class ReportAnalysisRequest(BaseRequest):
    report_text: str
    patient_id: Optional[str]
    report_type: Optional[str]  # "radiology" | "discharge_summary" | "lab_report"

class ReportAnalysisResponse(BaseResponse):
    patient_id: Optional[str]
    conditions: List[ExtractedCondition]   # each has: name, icd10_code, confidence, severity, negated
    raw_entities: List[str]
    model_version: str
```

### `drug.py` — cross-module schema (imports from both xray + report)
```python
class DrugRecommendationRequest(BaseRequest):
    patient_id: Optional[str]
    xray_label: Optional[XRayLabel]           # from xray module output
    xray_confidence: Optional[float]
    conditions: List[ExtractedCondition]      # from report module output
    patient_age: Optional[int]
    patient_weight_kg: Optional[float]
    known_allergies: List[str]
    current_medications: List[str]

class DrugRecommendationResponse(BaseResponse):
    recommendations: List[DrugRecommendation]   # drug_name, dosage, route, contraindications
    interactions_detected: List[DrugInteraction]
    disclaimer: str
```

### `tumor.py`
```python
class TumorSegmentationRequest(BaseRequest):
    image_base64: Optional[str]              # single 2D slice
    volume_slices_base64: Optional[List[str]] # 3D volume
    modality: "mri" | "ct"
    patient_id: Optional[str]
    body_region: Optional[str]              # "brain" | "lung" | "liver"

class TumorSegmentationResponse(BaseResponse):
    findings: List[TumorFinding]             # tumor_type, confidence, bounding_boxes, mask
    num_tumors_detected: int
    disclaimer: str
```

---

## API Endpoint Map

```
GET  /health                    → gateway health
GET  /xray/health               → xray module health
POST /xray/predict              → XRayRequest → XRayResponse
GET  /report/health             → report module health
POST /report/analyze            → ReportAnalysisRequest → ReportAnalysisResponse
GET  /drug/health               → drug module health
POST /drug/recommend            → DrugRecommendationRequest → DrugRecommendationResponse
GET  /tumor/health              → tumor module health
POST /tumor/segment             → TumorSegmentationRequest → TumorSegmentationResponse
```

**Typical orchestration flow:**
```
POST /xray/predict         →  get { label: "PNEUMONIA", confidence: 0.91 }
POST /report/analyze       →  get { conditions: [{ condition_name: "pneumonia", icd10: "J18.9" }] }
POST /drug/recommend       →  pass both above → get drug list + interactions
```

---

## Gateway (`gateway/main.py`)
- FastAPI app with lifespan context manager
- CORS middleware (allow all for dev, tighten for prod)
- Global exception handler → returns `ErrorResponse`
- Process time header middleware
- `router_registry.py` conditionally mounts each router based on `enable_xray` / `enable_report` etc. in `.env`
- Run: `uvicorn gateway.main:app --reload --host 0.0.0.0 --port 8000`
- Auto docs: `http://localhost:8000/docs`

---

## Git Workflow — 2 Devs

**Branch strategy:**
```
main       ← stable only, tagged releases
develop    ← both devs merge features here

feature/userA-xray-model
feature/userA-drug-engine
feature/userA-tumor-seg
feature/userB-report-nlp
shared/update-schemas      ← changes to shared/ — BOTH devs review
```

**Rules:**
- Never push directly to `main` or `develop`
- `shared/` changes always go through PR — both devs must approve
- `modules/xray_detection/`, `modules/drug_recommendation/`, `modules/tumor_segmentation/` → User A owns
- `modules/report_analyzer/` → User B owns
- `gateway/router_registry.py` → coordinate before editing (both touch it)

**Commit convention:**
```
feat(xray): add ResNet18 inference pipeline
fix(drug): handle empty conditions list
refactor(shared): extract image validation
test(report): add NER extraction unit tests
```

---

## `.gitignore` Key Entries
```
__pycache__/  *.pth  *.pt  *.ckpt  *.pkl  *.h5  *.onnx
data/models/  data/datasets/  data/outputs/
.env  *.env  venv/  .venv/
.idea/  .vscode/  .DS_Store
.pytest_cache/  htmlcov/  .coverage
```

---

## Files to Create (in order)

1. `.gitignore`
2. `requirements/base.txt`
3. `requirements/xray_detection.txt`
4. `requirements/report_analyzer.txt`
5. `requirements/drug_recommendation.txt`
6. `requirements/tumor_segmentation.txt`
7. `.env.example`
8. `shared/schemas/common.py`    ← must exist before all other schemas
9. `shared/schemas/xray.py`
10. `shared/schemas/report.py`
11. `shared/schemas/drug.py`     ← imports from xray + report
12. `shared/schemas/tumor.py`
13. `shared/schemas/__init__.py`
14. `shared/utils/config.py`
15. `shared/utils/image_utils.py`
16. `shared/utils/logger.py`
17. `shared/utils/exceptions.py`
18. `shared/utils/__init__.py`
19. `shared/__init__.py`
20. `gateway/main.py`
21. `gateway/router_registry.py`
22. `gateway/__init__.py`
23. `modules/xray_detection/router.py`
24. `modules/xray_detection/model.py`
25. `modules/xray_detection/service.py`
26. `modules/xray_detection/__init__.py`
27. `modules/report_analyzer/` (same 4 files)
28. `modules/drug_recommendation/` (same 4 files)
29. `modules/tumor_segmentation/` (same 4 files)
30. `scripts/download_models.py`
31. `scripts/setup_dev.sh`
32. `data/.gitkeep` + subdirs
33. `README.md`
34. `__init__.py` at repo root

---

## Verification

1. `pip install -r requirements/base.txt` — no errors
2. `uvicorn gateway.main:app --reload` — server starts, all 4 module routers register
3. `http://localhost:8000/docs` — all endpoints visible with correct schema
4. `pytest modules/ -v` — all tests pass
5. `POST /xray/predict` with sample base64 X-ray → valid `XRayResponse` JSON
6. `POST /report/analyze` with sample report text → valid `ReportAnalysisResponse`
7. `POST /drug/recommend` passing xray_label + conditions → valid `DrugRecommendationResponse`
8. `POST /tumor/segment` with base64 MRI slice → valid `TumorSegmentationResponse`
