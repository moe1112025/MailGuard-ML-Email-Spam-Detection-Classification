from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "email.csv"
ARTIFACTS = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS / "spam_pipeline.pkl"
REPORT_PATH = ARTIFACTS / "model_report.json"
RANDOM_STATE = 42


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    frame = pd.read_csv(DATA_PATH)
    required = {"category", "message"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame = frame[["category", "message"]].dropna().copy()
    frame["category"] = frame["category"].astype(str).str.strip().str.lower()
    frame["message"] = frame["message"].astype(str).str.strip()
    frame = frame[frame["category"].isin(["ham", "spam"])]
    frame = frame[frame["message"].ne("")]
    frame = frame.drop_duplicates(["category", "message"]).reset_index(drop=True)
    if frame["category"].nunique() != 2:
        raise ValueError("The dataset must contain ham and spam classes")
    return frame


def make_vectorizer():
    return TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True, max_features=10000)


def build_model_space():
    return {
        "Naive Bayes": (Pipeline([("tfidf", make_vectorizer()), ("model", MultinomialNB())]), {"model__alpha": [0.1, 0.5, 1.0, 2.0]}),
        "Logistic Regression": (Pipeline([("tfidf", make_vectorizer()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))]), {"model__C": [0.1, 1.0, 10.0]}),
        "Linear SVM": (Pipeline([("tfidf", make_vectorizer()), ("model", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE))]), {"model__C": [0.1, 1.0, 10.0]}),
        "Random Forest": (Pipeline([("tfidf", make_vectorizer()), ("model", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]), {"model__n_estimators": [150, 250], "model__max_depth": [None, 20]}),
        "Decision Tree": (Pipeline([("tfidf", make_vectorizer()), ("model", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE))]), {"model__max_depth": [10, 20, None]}),
    }


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    scores = model.decision_function(X_test) if hasattr(model, "decision_function") else model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "classification_report": classification_report(y_test, predictions, target_names=["HAM", "SPAM"], output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def train():
    frame = load_dataset()
    X = frame["message"]
    y = frame["category"].map({"ham": 0, "spam": 1}).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    folds = max(2, min(5, int(y_train.value_counts().min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    report = {}
    fitted = {}
    for name, (pipeline, params) in build_model_space().items():
        search = GridSearchCV(pipeline, params, scoring="f1", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        report[name] = {"best_params": search.best_params_, "cv_best_f1": float(search.best_score_), "metrics": evaluate(search.best_estimator_, X_test, y_test)}
    best_name = max(report, key=lambda name: report[name]["metrics"]["f1"])
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(fitted[best_name], file, protocol=pickle.HIGHEST_PROTOCOL)
    REPORT_PATH.write_text(json.dumps({"best_model": best_name, "dataset_size": len(frame), "models": report}, indent=2), encoding="utf-8")
    print(f"Best model: {best_name}")
    print(json.dumps(report[best_name]["metrics"], indent=2))


if __name__ == "__main__":
    train()
