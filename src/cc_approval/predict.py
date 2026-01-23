from __future__ import annotations
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from .config import paths
from .features import ALL_FEATURES

def load_model():
    return joblib.load(paths.model_path)

def predict_one(payload: Dict[str, Any]) -> Dict[str, float | int | bool]:
    # Ensure all expected features exist
    row = {k: payload.get(k, None) for k in ALL_FEATURES}
    df = pd.DataFrame([row])
    model = load_model()
    proba = float(model.predict_proba(df)[0, 1])
    approve = bool(proba >= 0.5)
    return {"approval_probability": proba, "approve": approve}
