from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .schemas import Application, Prediction
from .predict import predict_one

# Get the path to the static directory
static_dir = Path(__file__).parent / "static"

app = FastAPI(
    title="Credit Card Approval API",
    description="Machine Learning-powered credit card approval prediction system",
    version="0.1.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# Mount static files if directory exists
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=Prediction)
def predict(app_in: Application):
    result = predict_one(app_in.model_dump())
    return result
