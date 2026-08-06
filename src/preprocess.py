"""Data loading + the feature pipeline. Shared by training and serving."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET = "Churn"
DROP = ["customerID"]


def load_data(csv_path):
    """Read the CSV, fix TotalCharges, split into features X and target y (0/1)."""
    df = pd.read_csv(csv_path)
    # brand-new customers have a blank TotalCharges -> make numeric, fill with 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def build_preprocessor(X):
    """Scale numbers, one-hot encode categories. handle_unknown='ignore'
    so a new category at prediction time never crashes the API."""
    num = X.select_dtypes(include="number").columns.tolist()
    # exclude="number" (rather than include="object") keeps this working on
    # pandas 3.x, where text columns can arrive as the new `str` dtype.
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )
