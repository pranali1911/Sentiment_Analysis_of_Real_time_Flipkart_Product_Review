# sentiment_mlflow.py
# -----------------------
# Same structure as iris_mlflow.py
# - Optuna hyperparameter tuning (multiple models)
# - MLflow tracking + nested runs
# - Final training + test evaluation
# -----------------------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
# pip install optuna-integration[mlflow]
from optuna.integration.mlflow import MLflowCallback

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

import joblib
import time
import os
import re
import nltk
from nltk.corpus import stopwords
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # or 1
warnings.filterwarnings("ignore")

nltk.download("stopwords", quiet=True)


# -----------------------
# Step 1 - Load the dataset
# -----------------------
df = pd.read_csv("data.csv")

# -----------------------
# Step 2 - Data Cleaning + Label creation
# -----------------------
# dealing with missing values
# Remove rows with missing values in critical columns
df_cleaned = df.dropna(
    subset=["Review text", "Ratings"]
).copy()   # ✅ make an explicit copy

# Keep only valid ratings
df_cleaned = df_cleaned[df_cleaned["Ratings"].isin([1, 2, 3, 4, 5])]

# Remove neutral reviews (3 stars)
df_cleaned = df_cleaned[df_cleaned["Ratings"] != 3]

# Create sentiment label (4-5 => Positive(1), 1-2 => Negative(0))
df_cleaned["Sentiment"] = df_cleaned["Ratings"].apply(lambda x: 1 if x >= 4 else 0)


# -----------------------
# Step 3 - Text preprocessing
# -----------------------
stop_words = set(stopwords.words("english"))

def text_cleaning(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df_cleaned["clean_text"] = df_cleaned["Review text"].apply(text_cleaning)

# Remove empty cleaned reviews if any
df_cleaned = df_cleaned[df_cleaned["clean_text"].str.len() > 0].copy()

X = df_cleaned["clean_text"].astype(str)
y = df_cleaned["Sentiment"].astype(int)

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# -----------------------
# Step 4 - Define Pipeline (same style as IRIS)
# -----------------------
pipeline = Pipeline(
    [
        ("Vectorizer", TfidfVectorizer(stop_words="english")),
        ("Model", LogisticRegression())
    ]
)


# -----------------------
# Step 5 - Optuna Objective functions (CV on Train)
# -----------------------
def objective_logreg(trial):
    pipeline.set_params(
        Vectorizer=TfidfVectorizer(
            stop_words="english",
            ngram_range=trial.suggest_categorical("ngram_range", [(1, 1), (1, 2)]),
            max_features=trial.suggest_int("tfidf_max_features", 3000, 20000, step=1000),
            min_df=trial.suggest_int("min_df", 1, 5),
            max_df=trial.suggest_float("max_df", 0.80, 0.99),
        ),
        Model=LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=trial.suggest_float("C", 1e-3, 1e2, log=True),
            solver=trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
        )
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(pipeline, X_train, y_train, scoring="f1", cv=skf).mean()
    return score


def objective_linearsvc(trial):
    pipeline.set_params(
        Vectorizer=TfidfVectorizer(
            stop_words="english",
            ngram_range=trial.suggest_categorical("ngram_range", [(1, 1), (1, 2)]),
            max_features=trial.suggest_int("tfidf_max_features", 3000, 20000, step=1000),
            min_df=trial.suggest_int("min_df", 1, 5),
            max_df=trial.suggest_float("max_df", 0.80, 0.99),
        ),
        Model=LinearSVC(
            class_weight="balanced",
            C=trial.suggest_float("C", 1e-3, 1e2, log=True)
        )
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(pipeline, X_train, y_train, scoring="f1", cv=skf).mean()
    return score


def objective_svc_rbf(trial):
    pipeline.set_params(
        Vectorizer=TfidfVectorizer(
            stop_words="english",
            ngram_range=trial.suggest_categorical("ngram_range", [(1, 1), (1, 2)]),
            max_features=trial.suggest_int("tfidf_max_features", 3000, 20000, step=1000),
            min_df=trial.suggest_int("min_df", 1, 5),
            max_df=trial.suggest_float("max_df", 0.80, 0.99),
        ),
        Model=SVC(
            kernel="rbf",
            class_weight="balanced",
            C=trial.suggest_float("C", 1e-3, 1e2, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
        )
    )
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    score = cross_val_score(pipeline, X_train, y_train, scoring="f1", cv=skf).mean()
    return score


def objective_rf(trial):
    pipeline.set_params(
        Vectorizer=TfidfVectorizer(
            stop_words="english",
            ngram_range=trial.suggest_categorical("ngram_range", [(1, 1), (1, 2)]),
            max_features=trial.suggest_int("tfidf_max_features", 3000, 20000, step=1000),
            min_df=trial.suggest_int("min_df", 1, 5),
            max_df=trial.suggest_float("max_df", 0.80, 0.99),
        ),
        Model=RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 5, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            random_state=42,
            
        )
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(pipeline, X_train, y_train, scoring="f1", cv=skf).mean()
    return score


def objective_mnb(trial):
    # MultinomialNB works best on counts, but TF-IDF is also okay.
    pipeline.set_params(
        Vectorizer=TfidfVectorizer(
            stop_words="english",
            ngram_range=trial.suggest_categorical("ngram_range", [(1, 1), (1, 2)]),
            max_features=trial.suggest_int("tfidf_max_features", 3000, 20000, step=1000),
            min_df=trial.suggest_int("min_df", 1, 5),
            max_df=trial.suggest_float("max_df", 0.80, 0.99),
        ),
        Model=MultinomialNB(
            alpha=trial.suggest_float("alpha", 1e-3, 10.0, log=True)
        )
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(pipeline, X_train, y_train, scoring="f1", cv=skf).mean()
    return score


# Map model names to objective functions
objectives = {
    "LogisticRegression": objective_logreg,
    "LinearSVC": objective_linearsvc,
    "SVC_RBF": objective_svc_rbf,
    "RandomForest": objective_rf,
    "MultinomialNB": objective_mnb
}


# -----------------------
# Step 6 - MLflow Experiment
# -----------------------
mlflow.set_experiment("SENTIMENT_Analysis_RUNS")

results = {}
model_dict = {}
for i, model_name in enumerate(objectives.keys()):
    model_dict[model_name] = i


# -----------------------
# Step 7 - Loop through each algorithm (same style as IRIS)
# -----------------------
for model_name, obj_fn in objectives.items():
    print(f"\n--- Optimizing {model_name} ---")

    mlflow_cb = MLflowCallback(
        tracking_uri=None,            # Auto-detect
        metric_name="cv_f1",          # Primary metric
        mlflow_kwargs={
            "nested": True            # Child runs under parent
        }
    )

    # Create Optuna study
    study = optuna.create_study(direction="maximize")

    # Tune with Optuna
    start_fit = time.time()
    study.optimize(obj_fn, n_trials=20, callbacks=[mlflow_cb])
    fit_time = time.time() - start_fit

    best_params = study.best_params
    best_cv_f1 = study.best_value

    print(f"Best CV F1 for {model_name}: {best_cv_f1:.4f}")
    results[model_name] = {"best_params": best_params, "best_cv_f1": best_cv_f1}

    # Rebuild pipeline with best params (Vectorizer + Model)
    # Vectorizer params
    vec_params = {
        "stop_words": "english",
        "ngram_range": best_params["ngram_range"],
        "max_features": best_params["tfidf_max_features"],
        "min_df": best_params["min_df"],
        "max_df": best_params["max_df"]
    }
    pipeline.set_params(Vectorizer=TfidfVectorizer(**vec_params))

    # Model params
    if model_name == "LogisticRegression":
        pipeline.set_params(
            Model=LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=best_params["C"],
                solver=best_params["solver"]
            )
        )
    elif model_name == "LinearSVC":
        pipeline.set_params(
            Model=LinearSVC(
                class_weight="balanced",
                C=best_params["C"]
            )
        )
    elif model_name == "SVC_RBF":
        pipeline.set_params(
            Model=SVC(
                kernel="rbf",
                class_weight="balanced",
                C=best_params["C"],
                gamma=best_params["gamma"]
            )
        )
    elif model_name == "RandomForest":
        pipeline.set_params(
            Model=RandomForestClassifier(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                min_samples_split=best_params["min_samples_split"],
                min_samples_leaf=best_params["min_samples_leaf"],
                max_features=best_params["tfidf_max_features"],
                bootstrap=best_params["bootstrap"],
                random_state=42,
                n_jobs=-1
            )
        )
    elif model_name == "MultinomialNB":
        pipeline.set_params(
            Model=MultinomialNB(alpha=best_params["alpha"])
        )

    # Train final model
    pipeline.fit(X_train, y_train)

    # Evaluate on test data
    start_test = time.time()
    y_pred = pipeline.predict(X_test)
    test_time = time.time() - start_test

    # Metrics
    train_pred = pipeline.predict(X_train)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, y_pred)

    train_prec = precision_score(y_train, train_pred, zero_division=0)
    test_prec = precision_score(y_test, y_pred, zero_division=0)

    train_rec = recall_score(y_train, train_pred, zero_division=0)
    test_rec = recall_score(y_test, y_pred, zero_division=0)

    train_f1 = f1_score(y_train, train_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"{model_name} Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")
    print(f"{model_name} Fit Time: {fit_time:.2f}s, Test Time: {test_time:.2f}s")

    # Save model manually to track model size
    model_path = f"{model_name}_final_model.pkl"
    joblib.dump(pipeline, model_path)
    model_size = os.path.getsize(model_path)

    # Log in MLflow
    mlflow.log_metric("model_id", model_dict[model_name])
    mlflow.log_metric("cv_f1", best_cv_f1)

    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)

    mlflow.log_metric("train_precision", train_prec)
    mlflow.log_metric("test_precision", test_prec)

    mlflow.log_metric("train_recall", train_rec)
    mlflow.log_metric("test_recall", test_rec)

    mlflow.log_metric("train_f1", train_f1)
    mlflow.log_metric("test_f1", test_f1)

    mlflow.log_metric("train_time", fit_time)
    mlflow.log_metric("test_time", test_time)
    mlflow.log_metric("model_size", model_size)

    mlflow.sklearn.log_model(pipeline, name=f"{model_name}_sentiment_model")

    os.remove(model_path)

    results[model_name].update({
        "train_f1": train_f1,
        "test_f1": test_f1,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "fit_time": fit_time,
        "test_time": test_time,
        "model_size_bytes": model_size
    })

    mlflow.end_run()


# -----------------------
# Step 8 - Summary
# -----------------------
print("\n--- Summary ---")
for model_name, res in results.items():
    print(
        f"{model_name}: CV F1={res['best_cv_f1']:.4f}, "
        f"Train F1={res['train_f1']:.4f}, Test F1={res['test_f1']:.4f}, "
        f"Test Acc={res['test_accuracy']:.4f}, Fit Time={res['fit_time']:.2f}s, "
        f"Model Size={res['model_size_bytes']} bytes"
    )
