# Credit Card / ATM Fraud Detection

A Streamlit machine-learning demo for exploring synthetic transaction risk. It includes an overview dashboard, an individual transaction scorer, model comparison, and analytics.

> **Data notice:** Synthetic data is generated locally by default. The project also accepts the standard 284,807-row `Time`, `Amount`, `V1`–`V28`, `Class` CSV and transaction-record CSVs with `TransactionDate`, `Amount`, `MerchantID`, `TransactionType`, `Location`, `IsFraud`. The standard dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud); it is not bundled here. Synthetic model scores are educational examples, not evidence of production performance.

## Run locally

Use Python 3.10 or newer. From the project root, create and activate a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Prepare a quick 20,000-row synthetic dataset and train the comparison models:

```powershell
python -m scripts.prepare_demo
```

Start the application:

```powershell
streamlit run app.py
```

The app also offers a first-run setup button if the dataset and model bundle are missing. The **Data & Models** page accepts either supported CSV schema, validates it, then retrains all four models. Your `data/credit_card_fraud_dataset.csv` is supported directly; load and train it from PowerShell with `python -m scripts.prepare_demo --input data/credit_card_fraud_dataset.csv`. For transaction-record data, the app engineers model features from date/time, amount, merchant ID, transaction type, and location, and maps `IsFraud` to `Class` in memory. The original CSV is used directly and remains unchanged; the model-ready conversion is not written as a second CSV. Full-size training takes longer and requires more memory. To approximate the standard dataset row count with synthetic data, use `python -m scripts.prepare_demo --rows 284807`.

## What is included

- Imbalanced synthetic transaction generation with `Time`, `Amount`, `V1`–`V28`, and binary `Class` columns.
- Class-weighted Logistic Regression, calibrated linear SVM, Random Forest, and XGBoost models, evaluated on a stratified holdout split.
- Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, and model feature importance.
- Cached Streamlit views for dashboard KPIs, manual predictions, comparisons, and downloadable filtered analytics.
- Docker image, GitHub Actions tests, and an optional Render deploy-hook workflow.
- Tests for deterministic generation, schema validation, and invalid input handling.
- Docker image, GitHub Actions tests, and an optional Render deploy-hook workflow.

The trained model bundle (`models/model_bundle.joblib`) is excluded from version control. The app uses one dataset CSV at `data/credit_card_fraud_dataset.csv`.

## Test

```powershell
pytest
```

## Docker

```powershell
docker build -t fraud-detection-demo .
docker run --rm -p 8501:8501 fraud-detection-demo
```

Then open `http://localhost:8501` in a browser. The image prepares a small synthetic sample during build.

## Deploy

The included `render.yaml` describes a Render web service. Connect the repository in Render and deploy from the blueprint. The Docker build installs `requirements.txt` and prepares a 12,000-row synthetic sample; the start command launches Streamlit on Render's assigned port. To trigger deployment from GitHub Actions on pushes to `main`, add the Render service deploy-hook URL as the repository secret `RENDER_DEPLOY_HOOK`. The workflow skips deployment when the secret is absent. Uploaded datasets are stored in the app container's local filesystem; on free ephemeral hosting, re-upload after a redeploy or restart.

## Project layout

```text
app.py                         Streamlit interface
src/data.py                    Synthetic data generation and validation
src/modeling.py                Training, evaluation, and serialization
scripts/prepare_demo.py        Reproducible data/model preparation command
tests/test_project.py          Focused unit tests
.github/workflows/ci.yml       Test workflow
.github/workflows/deploy.yml   Optional Render deploy-hook workflow
Dockerfile                     Container deployment
render.yaml                    Render deployment blueprint
```