# Telecom Churn — a full MLOps pipeline

Predicts which telecom customers are about to leave, then serves that model as an API,
puts a web app in front of it, tests it on every push, and retrains it on a schedule.

The point of this project is not the model score — it's the **lifecycle around it**:
tracking, serving, testing, and automation.

```
data/telco_churn.csv ──> src/train.py ──> models/model.pkl ──> api/main.py ──> dashboard.py
                            │  (MLflow)                          (FastAPI)      (Streamlit)
                            └────────── GitHub Actions: test on push · retrain weekly · deploy
```

## The dataset

The public **IBM Telco Customer Churn** dataset — 7,043 real customers, 21 columns,
one row per customer with a known "did they leave" label.

| | |
|---|---|
| Rows | 7,043 |
| Features | 19 used (`customerID` dropped, `Churn` is the target) |
| Target | `Churn` → 1 if "Yes" |
| Class balance | **26.5% churn** — imbalanced, so accuracy is a misleading metric |

One data quirk matters: brand-new customers have a **blank** `TotalCharges`. It arrives as
text, so it is coerced to numeric and filled with `0.0` in `load_data()`.

## Results (measured, not estimated)

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| **Logistic regression** ← selected | **0.8416** | 0.6327 | 0.6136 |
| Random forest | 0.8227 | 0.6131 | 0.5372 |

Sanity checks through the live API:

| Customer | Churn probability |
|---|---|
| New, month-to-month, fiber, electronic check | **89.9% — high** |
| 70-month tenure, two-year contract, auto-pay, DSL | **3.4% — low** |
| Missing a required field | HTTP `422`, rejected before the model sees it |

Logistic regression beating a 300-tree random forest is not a mistake — churn here is
close to linearly separable in the one-hot space, and the forest overfits the majority
class despite `class_weight="balanced"`.

## Run it locally

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. get the data
mkdir -p data
curl -L -o data/telco_churn.csv \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# 2. train + track
python src/train.py
mlflow ui                            # http://127.0.0.1:5000 — compare the two runs

# 3. serve  (terminal 1)
uvicorn main:app --app-dir api --reload    # http://127.0.0.1:8000/docs

# 4. web app  (terminal 2)
streamlit run dashboard.py           # http://127.0.0.1:8501

# 5. quality gate
pytest -v
```

Try the API from the command line:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":1,
       "PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
       "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
       "StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month",
       "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
       "MonthlyCharges":95.0,"TotalCharges":95.0}'
```

## Run the API in Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Deploy (free, no card)

- **API → Render.** New Web Service → point at this repo → Docker. Render injects `$PORT`;
  the Dockerfile already reads it. Auto-deploys on every push to `main`.
- **UI → Streamlit Community Cloud.** Point at `dashboard.py` and set the environment
  variable `API_URL` to your Render URL.

## What each file is for

| File | Role |
|---|---|
| `src/preprocess.py` | The one feature pipeline, shared by training **and** serving |
| `src/train.py` | Trains 2 models, logs both to MLflow, saves the winner |
| `api/schema.py` | Typed input contract — Pydantic rejects bad requests |
| `api/main.py` | `/predict` and `/health` |
| `dashboard.py` | Streamlit form that calls the API |
| `tests/test_pipeline.py` | The quality gate CI runs on every push |
| `.github/workflows/ci.yml` | Runs the gate on push and PR |
| `.github/workflows/retrain.yml` | Weekly retrain → re-test → commit the new model |

## The five concepts to be ready to explain

1. **Class imbalance** — only 26.5% churn, so accuracy lies (predicting "nobody leaves"
   scores 73.5%). Judge on ROC-AUC and PR-AUC; correct with `class_weight="balanced"`.
2. **No train/serve skew** — the `ColumnTransformer` lives *inside* `model.pkl`, so the API
   cannot prepare data differently from training. `handle_unknown="ignore"` means a category
   never seen in training returns a score instead of a 500.
3. **Experiment tracking** — MLflow records params and metrics per run, so "logreg won" is
   evidence, not a claim.
4. **Quality gate in CI** — `test_pipeline_trains_and_beats_baseline` fails the build if
   ROC-AUC drops under 0.78. A worse model cannot reach production silently.
5. **Automated retrain** — a schedule retrains, re-tests, and commits the model, which
   triggers the deploy. That closed loop is what "MLOps" means.

## Two fixes applied to the build guide

- **Dockerfile model path.** The container's final `WORKDIR` is `/app/api`, so the default
  relative `models/model.pkl` resolves to `/app/api/models/model.pkl` and the API crashes on
  boot with `FileNotFoundError`. Fixed with `ENV MODEL_PATH=/app/models/model.pkl`.
- **Retrain workflow permissions.** The default `GITHUB_TOKEN` is read-only, so the
  final `git push` fails with a 403. Fixed with `permissions: contents: write`.
