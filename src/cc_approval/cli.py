from __future__ import annotations
import json
from pathlib import Path
import typer
import numpy as np
import pandas as pd

from .config import paths
from .train import train as train_fn
from .data_prep import load_data, split_data, save_splits, TARGET_COL
from .features import ALL_FEATURES
from .predict import predict_one

app = typer.Typer(help="Credit Card Approval CLI")

@app.command()
def generate_data(n: int = 1000, seed: int = 42) -> None:
    """Generate a synthetic dataset to data/raw/credit_card_applications.csv"""
    rng = np.random.default_rng(seed)
    genders = np.array(["male", "female"])
    educ_levels = np.array(["high_school", "bachelor", "master", "phd"])
    marital_status = np.array(["single", "married", "divorced"])
    housing = np.array(["own", "rent", "mortgage"])

    age = rng.integers(18, 70, size=n)
    income = rng.normal(60000, 20000, size=n).clip(10000, None)
    years_emp = rng.integers(0, 40, size=n)
    credit_score = rng.integers(300, 850, size=n)
    exist_debt = rng.normal(8000, 6000, size=n).clip(0, None)
    req_limit = rng.normal(5000, 3000, size=n).clip(500, None)
    years_addr = rng.integers(0, 20, size=n)
    dependents = rng.integers(0, 5, size=n)

    dti = (exist_debt / (income + 1e-6)).clip(0, 2.0)

    has_default_history = rng.choice(["yes", "no"], size=n, p=[0.2, 0.8])
    gender = rng.choice(genders, size=n)
    edu = rng.choice(educ_levels, size=n, p=[0.35, 0.45, 0.18, 0.02])
    marital = rng.choice(marital_status, size=n, p=[0.45, 0.45, 0.10])
    house = rng.choice(housing, size=n, p=[0.4, 0.4, 0.2])

    # Latent approval score
    z = (
        0.015 * (credit_score - 300)
        + 0.000015 * (income - 20000)
        - 0.8 * (has_default_history == "yes").astype(int)
        - 1.2 * dti
        + 0.02 * (years_emp - 2)
        + 0.01 * (years_addr - 1)
        - 0.05 * dependents
        - 0.0001 * req_limit
        + rng.normal(0, 0.5, size=n)
    )
    proba = 1 / (1 + np.exp(-z))
    approved = (proba > 0.5).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": income.round(2),
        "years_employed": years_emp,
        "credit_score": credit_score,
        "existing_debt": exist_debt.round(2),
        "requested_limit": req_limit.round(2),
        "debt_to_income": dti.round(4),
        "years_at_address": years_addr,
        "dependents": dependents,
        "gender": gender,
        "education_level": edu,
        "marital_status": marital,
        "housing_status": house,
        "has_default_history": has_default_history,
        "approved": approved,
    })
    paths.data_raw.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.data_raw, index=False)
    typer.echo(f"Generated dataset -> {paths.data_raw} ({len(df)} rows)")

@app.command()
def train() -> None:
    metrics = train_fn()
    typer.echo(json.dumps(metrics, indent=2))

@app.command()
def evaluate() -> None:
    import json
    from sklearn.metrics import classification_report, roc_auc_score
    import pandas as pd
    from joblib import load
    from .config import paths
    from .features import ALL_FEATURES

    # Load splits and model
    train_path = paths.data_processed / "train.csv"
    test_path = paths.data_processed / "test.csv"
    model = load(paths.model_path)

    test_df = pd.read_csv(test_path)
    X_test = test_df[ALL_FEATURES]
    y_test = test_df["approved"].astype(int)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    report = classification_report(y_test, preds, output_dict=True)
    summary = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": report["accuracy"],
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],
        "f1_1": report["1"]["f1-score"],
    }
    print(json.dumps(summary, indent=2))

@app.command()
def predict_one_cmd(example: bool = False, json_path: Path | None = None) -> None:
    if example:
        from .examples import SAMPLE
        payload = SAMPLE
    else:
        assert json_path is not None and json_path.exists(), "Provide --json-path to a file"
        payload = json.loads(Path(json_path).read_text())
    result = predict_one(payload)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    app()
