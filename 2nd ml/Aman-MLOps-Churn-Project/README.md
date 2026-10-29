# Customer Churn Risk — MLOps Mini Project

A self-contained educational MLOps project that trains a customer churn classifier, compares candidate models, exposes predictions through FastAPI, and checks the code with pytest and GitHub Actions.

## Why this is different
This project uses a **binary classification** workflow rather than house-price regression. It creates deterministic synthetic subscription/customer data so the project can be trained and tested offline without downloading private or licensed data.

> **Important:** Synthetic data is for demonstrating the workflow only. Metrics do not represent performance on real customers and must not be used for real business decisions.

## Features
- Synthetic data generator with a reproducible random seed.
- Preprocessing for numeric and categorical columns.
- Logistic Regression and Random Forest model comparison.
- Accuracy, precision, recall, F1, and ROC-AUC metrics.
- Persisted model bundle in `artifacts/`.
- FastAPI `/predict` and `/health` endpoints.
- Docker container and GitHub Actions CI.
- Unit tests for data generation and training.

## Layout
```text
Aman-MLOps-Churn-Project/
├── .github/workflows/ci.yml
├── app/main.py
├── src/churn_ml/data.py
├── src/churn_ml/train.py
├── tests/
├── artifacts/.gitkeep
├── Dockerfile
├── requirements.txt
└── README.md
```

## Run on Windows PowerShell
Python 3.11 recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.churn_ml.train
python -m pytest -q
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

Example request:
```json
{
  "tenure_months": 18,
  "monthly_fee": 74.5,
  "support_tickets": 2,
  "contract_type": "month_to_month",
  "payment_method": "card",
  "has_streaming": true
}
```

## Docker
```powershell
docker build -t aman-churn-mlops .
docker run --rm -p 8000:8000 aman-churn-mlops
```

## Push to your GitHub
Create an empty repository, e.g. `Aman-MLOps-Churn-Project`, under your account, then open a terminal in this extracted folder:

```powershell
git init
git add .
git commit -m "Build customer churn MLOps pipeline"
git branch -M main
git remote add origin https://github.com/villanious1/Aman-MLOps-Churn-Project.git
git push -u origin main
```

If you sign in to GitHub, use the browser authentication prompt. Never share credentials or access tokens in chat.

## Next steps
Replace synthetic data with a properly licensed real dataset, document the data source, validate performance on a held-out real dataset, and consider monitoring and model-drift reporting.
