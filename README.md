# 💳 Credit Card Approval ML — Production-Ready Pipeline

**Repository:** [JOSORO20/employee-attrition-ml](https://github.com/JOSORO20/employee-attrition-ml)

This is my end-to-end machine learning project that predicts whether a credit card application gets **approved** or **denied**. Built with best practices—clean code, type hints, proper testing, and production-ready deployment.

## What's Inside 🎯

- ✅ **Clean, modular code** with full type hints
- ✅ **Complete ML pipeline** (data prep → features → training → evaluation)
- ✅ **FastAPI service** for real-time predictions with custom colorful UI
- ✅ **CLI tools** for training, evaluation, and predictions
- ✅ **Full test coverage** with pytest
- ✅ **Code quality** enforced with ruff + black + pre-commit
- ✅ **GitHub Actions CI** that validates every push
- ✅ **VS Code ready** with launch configs and tasks
- ✅ **Docker support** for containerized deployment

**💡 Quick Start:** Everything is ready to run immediately—a synthetic dataset is pre-generated under `data/raw/` so you can test the full pipeline right away.

---

## Getting Started 🚀

**Setup (2 minutes):**

```bash
# 1️⃣ Create virtual environment
python -m venv .venv

# Activate it
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2️⃣ Install everything
pip install -r requirements.txt

# 3️⃣ Train the model
python -m cc_approval.cli train

# 4️⃣ Check performance
python -m cc_approval.cli evaluate

# 5️⃣ Start the API server (http://localhost:8000)
uvicorn cc_approval.app:app --reload --port 8000

# 6️⃣ Make a prediction (in another terminal)
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

**Optional:** Generate fresh training data:
```bash
python -m cc_approval.cli generate-data --n 1500 --seed 42
```

---

## Project Structure

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
│  ├─ app.py                 # FastAPI app with custom UI
│  ├─ cli.py                 # Typer CLI
│  ├─ config.py              # Paths & settings
│  ├─ data_prep.py           # load/split/save helpers
│  ├─ features.py            # preprocessing pipeline
│  ├─ predict.py             # prediction utilities
│  ├─ schemas.py             # Pydantic models for API
│  ├─ train.py               # training script
│  ├─ utils.py               # misc utilities
│  └─ static/
│      └─ swagger_ui_custom.html  # colorful API docs
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

## VS Code Setup ⚙️

I've pre-configured everything for a smooth development experience:

**Launch Configurations** (press F5 to run):
- `Run API (FastAPI/Uvicorn)` — Start the API with debugging
- `Pytest` — Run tests with debug support

**Tasks** (Ctrl+Shift+B):
- `Install deps` — Install all dependencies
- `Train model` — Train the ML pipeline
- `Evaluate` — Check model performance
- `Serve API` — Start the API server

**Recommended Extensions:**
- Python (Microsoft)
- Pylance (type checking)
- Ruff (linting)

---

## How the Pipeline Works 🔄

I built this with a clear separation of concerns. Here's the flow:

1. **`data_prep.py`** → Loads & splits the dataset
2. **`features.py`** → Preprocesses data (scaling, encoding, feature engineering)
3. **`train.py`** → Trains a scikit-learn Pipeline with the model
4. **`predict.py`** → Loads the saved model and makes predictions
5. **`schemas.py`** → Validates API requests with Pydantic
6. **`app.py`** → FastAPI endpoint that ties it all together
7. **`cli.py`** → Command-line interface for training/evaluation

## Model Performance 📊

**Latest results on test set:**
- **ROC AUC: 0.977** ✨ (excellent discrimination)
- **Accuracy: 90.4%** (9 out of 10 correct)
- **Precision: 99.5%** (very few false approvals)
- **F1 Score: 0.948** (solid balance)

---

## Deploy with Docker 🐳

Quick containerized deployment:

```bash
docker build -t cc-approval-api .
docker run -p 8000:8000 cc-approval-api
```

---

## Code Quality 🎯

I use industry-standard tools to maintain high code quality:

- **Ruff** for linting
- **Black** for formatting
- **Pre-commit** hooks to catch issues before commits
- **Pytest** for unit tests
- **GitHub Actions** CI to validate every push

Install pre-commit hooks:
```bash
pre-commit install
```

Run tests locally:
```bash
pytest tests/ -v
```

---

## API Documentation 🎨

The API comes with a beautiful, interactive documentation at `http://localhost:8000/docs`. I created a custom Swagger UI with:
- 🎨 Modern gradient design (purple/violet theme)
- 🎯 Color-coded HTTP methods
- ✨ Smooth animations and transitions
- 📱 Responsive layout

Try out the `/predict` endpoint directly from the browser!

---

## Next Steps

- Clone this repo and run it locally
- Check out the model metrics—they're pretty solid!
- Customize the features in `features.py`
- Deploy to production using Docker
- Modify the API styling in `src/cc_approval/static/swagger_ui_custom.html`

---

**Built with:** Python • FastAPI • scikit-learn • Pydantic • Docker 🚀

Made with ❤️ by JOSORO20
