# Credit Card Approval — End-to-End ML Project

An end-to-end machine learning project that predicts whether a credit card application should be **approved**. It includes:
- Clean, modular **src** code with type hints
- **Training, evaluation, and inference** scripts
- A **FastAPI** service for real-time predictions
- **CLI** commands with Typer
- **Unit tests** with pytest
- **Pre-commit** (ruff + black) for code quality
- **GitHub Actions** CI
- Ready for **VS Code** (launch & tasks) and Docker

> NOTE: A small **synthetic dataset** is generated for you under `data/raw/` so you can run the whole pipeline immediately.

---

## 1) Quickstart

```bash
# 1) Create & activate a virtual env (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
# (optional packaging install:)
# pip install -e .

# 3) (Optional) Generate fresh synthetic data
python -m cc_approval.cli generate-data --n 1500 --seed 42

# 4) Train
python -m cc_approval.cli train

# 5) Evaluate (prints metrics)
python -m cc_approval.cli evaluate

# 6) Predict on one JSON record
python -m cc_approval.cli predict-one --example

# 7) Serve an API
uvicorn cc_approval.app:app --reload --port 8000

# 8) Call the API (separate terminal)
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @examples/sample_request.json
```

---

## 2) Project Structure

```
cc-approval-ml/
├─ .github/workflows/ci.yml
├─ .vscode/
│  ├─ launch.json
│  └─ tasks.json
├─ data/
│  ├─ raw/credit_card_applications.csv       # synthetic dataset
│  ├─ interim/.gitkeep
│  └─ processed/.gitkeep
├─ examples/
│  └─ sample_request.json
├─ models/.gitkeep
├─ notebooks/
│  └─ 01_quick_eda.ipynb
├─ src/cc_approval/
│  ├─ __init__.py
│  ├─ app.py                 # FastAPI app
│  ├─ cli.py                 # Typer CLI
│  ├─ config.py              # Paths & settings
│  ├─ data_prep.py           # load/split/save helpers
│  ├─ features.py            # preprocessing pipeline
│  ├─ predict.py             # prediction utilities
│  ├─ schemas.py             # Pydantic models for API
│  ├─ train.py               # training script
│  └─ utils.py               # misc utilities
├─ tests/
│  ├─ test_training.py
│  └─ test_predict.py
├─ .gitignore
├─ .pre-commit-config.yaml
├─ Dockerfile
├─ LICENSE
├─ Makefile
├─ pyproject.toml
├─ requirements.txt
└─ README.md
```

---

## 3) VS Code

- Open the folder in VS Code.
- Recommended extensions will show (Python, Pylance). You also have ready-made `tasks` and `launch` to run the API & tests.

---

## 4) Step-by-step (learning path)

1. **Skim the code layout** under `src/cc_approval/` to see how modules are separated.
2. **Data**: Check `data/raw/credit_card_applications.csv`. Then open `data_prep.py` to see how it's loaded and split.
3. **Features**: Study `features.py` for the preprocessing pipeline (ColumnTransformer for numeric/categorical).
4. **Model**: Look at `train.py` to see how we build and fit a Pipeline, save it with `joblib`.
5. **Evaluation**: See `evaluate.py` code paths inside CLI to compute metrics (AUC/F1/Accuracy).
6. **Prediction**: `predict.py` shows loading the persisted pipeline and returning probabilities.
7. **API**: `app.py` exposes `/predict` using `schemas.py` to validate input.
8. **CLI**: `cli.py` gives commands for generate-data/train/evaluate/predict.
9. **Quality**: `pre-commit` with `ruff` and `black`. Try `pre-commit install` then commit to see hooks.
10. **CI**: Inspect `.github/workflows/ci.yml` – it lints and runs tests on every push/PR.
11. **Docker**: Build and run the containerized API.
    ```bash
    docker build -t cc-approval-api .
    docker run -p 8000:8000 cc-approval-api
    ```

Happy learning and shipping! 🚀
