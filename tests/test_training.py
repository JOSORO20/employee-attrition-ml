from pathlib import Path
import json
from cc_approval.cli import generate_data
from cc_approval.train import train
from cc_approval.config import paths

def test_train_end_to_end(tmp_path, monkeypatch):
    # Redirect paths to a temp dir
    monkeypatch.setattr(paths, "root", tmp_path, raising=False)
    monkeypatch.setattr(paths, "data_raw", tmp_path / "data" / "raw" / "credit_card_applications.csv", raising=False)
    monkeypatch.setattr(paths, "data_processed", tmp_path / "data" / "processed", raising=False)
    monkeypatch.setattr(paths, "models_dir", tmp_path / "models", raising=False)
    monkeypatch.setattr(paths, "model_path", tmp_path / "models" / "model.joblib", raising=False)
    monkeypatch.setattr(paths, "metrics_path", tmp_path / "models" / "metrics.json", raising=False)

    generate_data(n=200, seed=1)
    metrics = train()
    assert 0.5 <= metrics["roc_auc"] <= 1.0
    assert Path(paths.model_path).exists()
    assert json.loads(Path(paths.metrics_path).read_text())
