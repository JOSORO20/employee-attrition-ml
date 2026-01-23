from cc_approval.cli import generate_data
from cc_approval.train import train
from cc_approval.predict import predict_one
from cc_approval.examples import SAMPLE
from cc_approval.config import paths

def test_predict_flow(tmp_path, monkeypatch):
    # Override paths
    monkeypatch.setattr(paths, "root", tmp_path, raising=False)
    monkeypatch.setattr(paths, "data_raw", tmp_path / "data" / "raw" / "credit_card_applications.csv", raising=False)
    monkeypatch.setattr(paths, "data_processed", tmp_path / "data" / "processed", raising=False)
    monkeypatch.setattr(paths, "models_dir", tmp_path / "models", raising=False)
    monkeypatch.setattr(paths, "model_path", tmp_path / "models" / "model.joblib", raising=False)
    monkeypatch.setattr(paths, "metrics_path", tmp_path / "models" / "metrics.json", raising=False)

    generate_data(n=200, seed=2)
    train()
    res = predict_one(SAMPLE)
    assert 0.0 <= res["approval_probability"] <= 1.0
    assert isinstance(res["approve"], bool)
