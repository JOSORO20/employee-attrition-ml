from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel

class Paths(BaseModel):
    root: Path = Path(__file__).resolve().parents[2]
    data_raw: Path = root / "data" / "raw" / "credit_card_applications.csv"
    data_processed: Path = root / "data" / "processed"
    models_dir: Path = root / "models"
    model_path: Path = models_dir / "model.joblib"
    metrics_path: Path = root / "models" / "metrics.json"

paths = Paths()
