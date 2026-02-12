import os
import time
import json
import joblib
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
import optuna
from optuna.integration.mlflow import MLflowCallback

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "data.csv"              # change if needed
TEXT_COL = "Review text"            # your column name
RATING_COL = "Ratings"              # your column name
EXPERIMENT_NAME = "FLIPKART_SENTIMENT_OPTUNA_MLFLOW"
RANDOM_STATE = 42
N_TRIALS = 25

os.environ["LOKY_MAX_CPU_COUNT"] = "4"


# ----------------------------
# DATA LOADING + BASIC CLEAN
# ----------------------------
df = pd.read_csv(DATA_PATH)

# keep valid ratings
df = df[df[RATING_COL].isin([1, 2, 3, 4, 5])].copy()

# remove neutral
df = df[df[RATING_COL] != 3].copy()

# sentiment label: 1 (>=4) positive, else 0
df["Sentiment"] = df[RATING_COL].apply(lambda x: 1 if x >= 4 else 0)

# text column safe
df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("")

# remove empty
df = df[df[TEXT_COL].str.strip().str.len() > 0].copy()

X = df[TEXT_COL]
y = df["Sentiment"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)


# ----------------------------
# HELPERS
# ----------------------------
def build_vectorizer(trial):
    vec_type = trial.suggest_categorical("vectorizer_type", ["tfidf", "bow"])
    ngram_range = trial.suggest_categorical("ngram_range", [(1, 1), (1, 2)])
    min_df = trial.suggest_int("min_df", 2, 5)
    max_df = trial.suggest_float("max_df", 0.85, 0.98)

    if vec_type == "tfidf":
        max_features = trial.suggest_int("max_features", 5000, 30000, step=5000)
        return TfidfVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features
        )
    else:
        max_features = trial.suggest_int("max_features", 5000, 30000, step=5000)
        return CountVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features
        )


def evaluate_on_test(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }

    # ROC-AUC (only if we can get scores)
    try:
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
            metrics["test_roc_auc"] = float(roc_auc_score(y_test, y_score))
        elif hasattr(pipeline.named_steps["model"], "decision_function"):
            y_score = pipeline.decision_function(X_test)
            metrics["test_roc_auc"] = float(roc_auc_score(y_test, y_score))
    except Exception:
        pass

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    return metrics, cm, report


def log_artifacts(cm, report, model_name):
    os.makedirs("artifacts", exist_ok=True)

    cm_path = f"artifacts/{model_name}_confusion_matrix.txt"
    rep_path = f"artifacts/{model_name}_classification_report.txt"

    with open(cm_path, "w", encoding="utf-8") as f:
        f.write(np.array2string(cm))

    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report)

    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(rep_path)


def save_and_log_model(pipeline, model_name):
    os.makedirs("models", exist_ok=True)
    path = f"models/{model_name}_pipeline.joblib"
    joblib.dump(pipeline, path)

    size_bytes = os.path.getsize(path)
    mlflow.log_metric("model_size_bytes", float(size_bytes))

    mlflow.log_artifact(path)
    mlflow.sklearn.log_model(pipeline, artifact_path=f"{model_name}_sklearn_model")

    return path, size_bytes


# ----------------------------
# OPTUNA OBJECTIVES PER MODEL
# ----------------------------
def objective_factory(model_name):
    def objective(trial):
        vectorizer = build_vectorizer(trial)

        # choose model hyperparams
        if model_name == "LogReg":
            C = trial.suggest_float("C", 0.1, 10.0, log=True)
            clf = LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=C,
                n_jobs=None
            )

        elif model_name == "LinearSVC":
            C = trial.suggest_float("C", 0.1, 10.0, log=True)
            clf = LinearSVC(class_weight="balanced", C=C, dual=False)

        elif model_name == "MultinomialNB":
            alpha = trial.suggest_float("alpha", 0.1, 2.0)
            clf = MultinomialNB(alpha=alpha)

        elif model_name == "KNN":
            n_neighbors = trial.suggest_int("n_neighbors", 3, 31, step=2)
            # cosine distance works with brute in many cases; keep it simple & stable
            clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")

        else:
            raise ValueError("Unknown model_name")

        pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("model", clf),
        ])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        score = cross_val_score(
            pipeline, X_train, y_train,
            scoring="f1_macro",
            cv=skf,
            n_jobs=-1
        ).mean()

        return score

    return objective


# ----------------------------
# RUN EXPERIMENT
# ----------------------------
mlflow.set_experiment(EXPERIMENT_NAME)

models_to_run = ["LogReg", "LinearSVC", "MultinomialNB", "KNN"]
results = {}

with mlflow.start_run(run_name="FLIPKART_PARENT_RUN"):
    mlflow.log_param("data_path", DATA_PATH)
    mlflow.log_param("text_col", TEXT_COL)
    mlflow.log_param("rating_col", RATING_COL)
    mlflow.log_param("n_trials", N_TRIALS)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_metric("n_rows", float(len(df)))
    mlflow.log_metric("pos_rate", float(y.mean()))

    for model_name in models_to_run:
        print(f"\n--- Optimizing {model_name} ---")

        # callback will create nested MLflow runs for each trial
        mlflow_cb = MLflowCallback(
            tracking_uri=None,
            metric_name="cv_f1_macro",
            mlflow_kwargs={"nested": True}
        )

        study = optuna.create_study(direction="maximize")
        start_opt = time.time()
        study.optimize(
            objective_factory(model_name),
            n_trials=N_TRIALS,
            callbacks=[mlflow_cb]
        )
        opt_time = time.time() - start_opt

        best_params = study.best_params
        best_cv = float(study.best_value)

        print("Best params:", best_params)
        print("Best CV F1(macro):", best_cv)

        # Rebuild BEST pipeline from best_params
        # --- Vectorizer ---
        if best_params["vectorizer_type"] == "tfidf":
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=best_params["ngram_range"],
                min_df=best_params["min_df"],
                max_df=best_params["max_df"],
                max_features=best_params["max_features"],
            )
        else:
            vectorizer = CountVectorizer(
                stop_words="english",
                ngram_range=best_params["ngram_range"],
                min_df=best_params["min_df"],
                max_df=best_params["max_df"],
                max_features=best_params["max_features"],
            )

        # --- Model ---
        if model_name == "LogReg":
            model = LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=best_params["C"]
            )
        elif model_name == "LinearSVC":
            model = LinearSVC(
                class_weight="balanced",
                C=best_params["C"],
                dual=False
            )
        elif model_name == "MultinomialNB":
            model = MultinomialNB(alpha=best_params["alpha"])
        else:  # KNN
            model = KNeighborsClassifier(
                n_neighbors=best_params["n_neighbors"],
                metric="cosine",
                algorithm="brute"
            )

        best_pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("model", model)
        ])

        # Log a child run for the final fit/eval of this model
        with mlflow.start_run(run_name=f"{model_name}_BEST", nested=True):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("best_params", json.dumps(best_params))
            mlflow.log_metric("best_cv_f1_macro", best_cv)
            mlflow.log_metric("optuna_time_sec", float(opt_time))

            # Fit
            start_fit = time.time()
            best_pipeline.fit(X_train, y_train)
            fit_time = time.time() - start_fit
            mlflow.log_metric("fit_time_sec", float(fit_time))

            # Train accuracy (quick)
            train_pred = best_pipeline.predict(X_train)
            mlflow.log_metric("train_accuracy", float(accuracy_score(y_train, train_pred)))

            # Test
            start_test = time.time()
            test_metrics, cm, report = evaluate_on_test(best_pipeline, X_test, y_test)
            test_time = time.time() - start_test
            mlflow.log_metric("test_time_sec", float(test_time))

            for k, v in test_metrics.items():
                mlflow.log_metric(k, float(v))

            log_artifacts(cm, report, model_name)

            # Save + log model
            model_path, model_size = save_and_log_model(best_pipeline, model_name)

            results[model_name] = {
                "best_cv_f1_macro": best_cv,
                "test_metrics": test_metrics,
                "fit_time_sec": fit_time,
                "test_time_sec": test_time,
                "model_size_bytes": model_size,
                "model_path": model_path,
                "best_params": best_params
            }

    # Print summary
    print("\n================ SUMMARY ================\n")
    for m, res in results.items():
        print(
            f"{m}: CV_F1={res['best_cv_f1_macro']:.4f} | "
            f"Test_F1macro={res['test_metrics'].get('test_f1_macro', np.nan):.4f} | "
            f"Test_Acc={res['test_metrics'].get('test_accuracy', np.nan):.4f} | "
            f"Fit={res['fit_time_sec']:.2f}s | Test={res['test_time_sec']:.2f}s | "
            f"Size={res['model_size_bytes']} bytes"
        )

print("\nDone. Open MLflow UI with: mlflow ui")
