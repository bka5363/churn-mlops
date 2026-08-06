"""CI tests: the data loads, the pipeline trains, and it predicts sanely.
These run in GitHub Actions on every push.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor, load_data

DATA = os.getenv("DATA_PATH", "data/telco_churn.csv")


def test_data_loads():
    X, y = load_data(DATA)
    assert len(X) == len(y) > 1000
    assert set(y.unique()) <= {0, 1}
    assert "customerID" not in X.columns


def test_pipeline_trains_and_beats_baseline():
    X, y = load_data(DATA)
    pre = build_preprocessor(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    pipe = Pipeline(
        [("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]
    )
    pipe.fit(Xtr, ytr)
    auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
    assert auc > 0.78, f"model regressed: ROC-AUC={auc:.3f}"


def test_high_risk_scores_above_low_risk():
    """A month-to-month fiber newcomer must score higher than a loyal two-year customer."""
    import pandas as pd

    X, y = load_data(DATA)
    pre = build_preprocessor(X)
    pipe = Pipeline(
        [("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]
    )
    pipe.fit(X, y)

    risky = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 95.0,
    }
    safe = dict(
        risky, tenure=70, Contract="Two year", InternetService="DSL",
        PaymentMethod="Credit card (automatic)", PaperlessBilling="No",
        OnlineSecurity="Yes", TechSupport="Yes", StreamingTV="No",
        StreamingMovies="No", MonthlyCharges=45.0, TotalCharges=3150.0,
    )

    p_risky = pipe.predict_proba(pd.DataFrame([risky]))[0, 1]
    p_safe = pipe.predict_proba(pd.DataFrame([safe]))[0, 1]
    assert p_risky > 0.5 > p_safe, f"sanity check failed: {p_risky:.3f} vs {p_safe:.3f}"


def test_unknown_category_does_not_crash():
    """handle_unknown='ignore' means a never-seen category must not raise."""
    import pandas as pd

    X, y = load_data(DATA)
    pre = build_preprocessor(X)
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)

    row = X.iloc[[0]].copy()
    row["Contract"] = "Lifetime deal"  # a category that never existed in training
    assert 0.0 <= float(pipe.predict_proba(row)[0, 1]) <= 1.0
