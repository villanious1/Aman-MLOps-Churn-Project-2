"""Train and evaluate candidate churn classifiers."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.churn_ml.data import CATEGORICAL, FEATURES, NUMERIC, TARGET, make_customer_data

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "churn_model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"


def build_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])


def train_and_evaluate(n_samples=1500, random_state=21):
    df = make_customer_data(n_samples, random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=random_state,
        stratify=df[TARGET],
    )
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=140, min_samples_leaf=3, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        ),
    }
    results, fitted = {}, {}
    for name, classifier in candidates.items():
        model = Pipeline([("preprocess", build_preprocessor()), ("classifier", classifier)])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
        }
        fitted[name] = model
    winner = max(results, key=lambda name: results[name]["f1"])
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted[winner], "features": FEATURES, "target": TARGET}, MODEL_PATH)
    report = {
        "data": "synthetic subscription data",
        "n_samples": n_samples,
        "selected_model": winner,
        "selection_metric": "highest F1 on held-out synthetic test split",
        "models": results,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))
