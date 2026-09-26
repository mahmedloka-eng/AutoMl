"""
Metrics, training, tuning, evaluation, saving, MLflow-safe logging.
Imports: config, model_moka only.
"""
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.base import clone
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error,
)

from config import Config
from model_moka import get_model_moka                     

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
#  MLflow (file store)
# -----------------------------------------------------------------------------
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true" #mlflow pro fix

try:
    import mlflow
    import mlflow.sklearn

    _EXP = Config.EXPERIMENT_DIR 
    mlflow.set_tracking_uri(f"file://{_EXP}")
    mlflow.set_experiment("automl")
    HAS_MLFLOW = True
    logger.info("MLflow file store → %s", _EXP)
except Exception as e:
    logger.warning("MLflow unavailable (%s) — skipping.", type(e).__name__)
    HAS_MLFLOW = False
# except Exception as e:
#     import traceback
#     logger.warning("MLflow unavailable (%s: %s) — skipping.", type(e).__name__, e)
#     traceback.print_exc()
#     HAS_MLFLOW = False (so i can see the problem)


# =============================================================================
# Metrics
# =============================================================================
def scoring_metric(problem_type: str) -> str:
    return "f1_weighted" if problem_type == "classification" else "r2"


def primary_score(metrics: dict, problem_type: str) -> float:
    if problem_type == "classification":
        return metrics.get("f1_weighted", metrics.get("accuracy", 0.0))
    return metrics.get("r2", 0.0)


def compute_metrics(y_true, y_pred, problem_type: str) -> dict:
    if problem_type == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        }
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


# =============================================================================
# Leaderboard
# =============================================================================
def build_leaderboard(results: List[dict], task_type: str) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "Model": r["name"],
            "Source": r["source"],
            "CV_Mean": round(r["cv_mean"], 4),
            "CV_Std": round(r["cv_std"], 4),
            "Val_Score": round(primary_score(r["val_metrics"], task_type), 4),
            "Time (s)": r.get("time_s", 0.0),
        }
        tm = r.get("test_metrics") or {}
        if task_type == "classification":
            row["Test_F1"] = round(tm.get("f1_weighted", 0.0), 4)
            row["Test_Acc"] = round(tm.get("accuracy", 0.0), 4)
        else:
            row["Test_R2"] = round(tm.get("r2", 0.0), 4)
            row["Test_RMSE"] = round(tm.get("rmse", 0.0), 4)
            row["Test_MAE"] = round(tm.get("mae", 0.0), 4)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values("CV_Mean", ascending=False)
            .reset_index(drop=True))


# =============================================================================
# MLflow-safe logging
# =============================================================================
def _safe_mlflow_log(model, name, train_m, val_m, cv_mean, cv_std,
                     stage="baseline", extra=None):
    """Log to MLflow. Never raises — logging failure must not kill the model."""
    if not HAS_MLFLOW:
        return
    try:
        with mlflow.start_run(run_name=f"{stage}_{name}"):
            mlflow.log_param("stage", stage)
            mlflow.log_param("model_name", name)

            if extra:
                for k in ("task_type", "target_column", "source"):
                    if extra.get(k) is not None:
                        mlflow.log_param(k, str(extra[k]))

            safe_params = {
                f"param_{k}": v
                for k, v in model.get_params().items()
                if isinstance(v, (int, float, str, bool))
            }
            mlflow.log_params(safe_params)

            for k, v in train_m.items():
                mlflow.log_metric(f"train_{k}", v)
            for k, v in val_m.items():
                mlflow.log_metric(f"val_{k}", v)
            mlflow.log_metric("cv_mean", cv_mean)
            mlflow.log_metric("cv_std", cv_std)

            if extra and extra.get("test_metrics"):
                for k, v in extra["test_metrics"].items():
                    mlflow.log_metric(f"test_{k}", float(v))

            try:
                mlflow.sklearn.log_model(
                    model, "model",
                    skops_trusted_types=["sklearn.tree._tree.Tree"],
                )
            except TypeError:
                mlflow.sklearn.log_model(model, "model")

            if extra and extra.get("preprocessor") is not None:
                try:
                    art_uri = mlflow.get_artifact_uri("preprocessor")
                    art_dir = Path(art_uri.replace("file://", ""))
                    art_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(extra["preprocessor"], art_dir / "preprocessor.pkl")
                except Exception as e:
                    logger.warning("Could not log preprocessor: %s", e)
    except Exception as e:
        logger.warning("MLflow logging failed for %s: %s", name, e)


# =============================================================================
# Baseline training
# =============================================================================
def train_models(X_train, y_train, X_val, y_val, task_type, model_names=None):
    if X_train is None:
        return None, None, None, "Run preprocessing first."

    moka = get_model_moka(task_type)                       
    if not model_names or "all" in model_names:
        selected = list(moka.keys())
    else:
        selected = [n for n in model_names if n in moka]
    if not selected:
        return None, None, None, "No valid models selected."

    scoring = scoring_metric(task_type)
    results, trained = [], {}

    for name in selected:
        estimator, _grid = moka[name]                      
        logger.info("Baseline training: %s", name)
        try:
            start = time.time()
            model = clone(estimator)
            model.fit(X_train, y_train)
            duration = time.time() - start

            train_m = compute_metrics(y_train, model.predict(X_train), task_type)
            val_m = compute_metrics(y_val, model.predict(X_val), task_type)
            cv = cross_val_score(
                model, X_train, y_train,
                cv=Config.CV_FOLDS, scoring=scoring, n_jobs=Config.N_JOBS,
            )

            _safe_mlflow_log(
                model, name, train_m, val_m,
                float(cv.mean()), float(cv.std()),
                stage="baseline",
                extra={"task_type": task_type, "source": "baseline"},
            )

            results.append({
                "name": name,
                "source": "baseline",
                "fitted_model": model,
                "train_metrics": train_m,
                "val_metrics": val_m,
                "cv_mean": float(cv.mean()),
                "cv_std": float(cv.std()),
                "time_s": round(duration, 2),
            })
            trained[name] = model
        except Exception:
            logger.exception("Baseline training failed for %s", name)

    if not results:
        return None, None, None, "No models trained successfully."
    return (build_leaderboard(results, task_type), results, trained,
            f"✅ Trained {len(results)} baseline models.")


# =============================================================================
# Tuning
# =============================================================================
def select_tuning_candidates(baseline_results, top_k=None):
    """Default: tune every baseline. top_k != None → top-K only."""
    if not baseline_results:
        return []
    if top_k is None:
        return list(baseline_results)
    ranked = sorted(baseline_results, key=lambda r: r["cv_mean"], reverse=True)
    return ranked[:top_k]


def tune_models(X_train, y_train, task_type, candidates):
    if not candidates:
        return []
    scoring = scoring_metric(task_type)
    records = []
    for cand in candidates:
        name = cand["name"]
        moka = get_model_moka(task_type)                   
        if name not in moka:
            continue
        estimator, space = moka[name]                      
        logger.info("Tuning %s …", name)
        try:
            search = RandomizedSearchCV(
                estimator=clone(estimator),
                param_distributions=space,
                n_iter=min(Config.N_ITER_SEARCH,
                           max(3, sum(len(v) for v in space.values()))),
                cv=Config.CV_FOLDS,
                scoring=scoring,
                n_jobs=Config.N_JOBS,
                random_state=Config.RANDOM_STATE,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            records.append({
                "name": name,
                "best_estimator": search.best_estimator_,
                "best_params": search.best_params_,
                "best_score": float(search.best_score_),
            })
            logger.info("  best %s: %.4f", name, search.best_score_)
        except Exception as e:
            logger.warning("Tuning failed for %s: %s", name, e)
    return records


def retrain_tuned(X_train, y_train, X_val, y_val, task_type, tuning_records):
    if not tuning_records:
        return []
    scoring = scoring_metric(task_type)
    results = []
    for rec in tuning_records:
        name = rec["name"] + " (tuned)"
        model = rec["best_estimator"]
        try:
            start = time.time()
            model.fit(X_train, y_train)
            duration = time.time() - start

            train_m = compute_metrics(y_train, model.predict(X_train), task_type)
            val_m = compute_metrics(y_val, model.predict(X_val), task_type)
            cv = cross_val_score(model, X_train, y_train,
                                 cv=Config.CV_FOLDS, scoring=scoring, n_jobs=Config.N_JOBS)

            _safe_mlflow_log(
                model, name, train_m, val_m,
                float(cv.mean()), float(cv.std()),
                stage="tuned",
                extra={"task_type": task_type, "source": "tuned"},
            )

            results.append({
                "name": name,
                "source": "tuned",
                "fitted_model": model,
                "train_metrics": train_m,
                "val_metrics": val_m,
                "cv_mean": float(cv.mean()),
                "cv_std": float(cv.std()),
                "time_s": round(duration, 2),
            })
        except Exception as e:
            logger.warning("Retrain failed for %s: %s", name, e)
    return results


# =============================================================================
# Evaluation + saving
# =============================================================================
def pick_final_winner(baseline_results, tuned_results):
    pool = list(baseline_results or []) + list(tuned_results or [])
    if not pool:
        raise ValueError("No trained models to pick from.")
    return max(pool, key=lambda r: r["cv_mean"])


def evaluate_test(model, X_test, y_test, task_type):
    """Returns (metrics_dict, matplotlib Figure)."""
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, task_type)

    fig, ax = plt.subplots(figsize=(6, 5))
    if task_type == "regression":
        ax.scatter(y_test, y_pred, alpha=0.5)
        lims = [min(np.min(y_test), np.min(y_pred)), max(np.max(y_test), np.max(y_pred))]
        ax.plot(lims, lims, "r--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Test: Actual vs Predicted")
    else:
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Test: Confusion Matrix")

    plt.tight_layout()
    return metrics, fig


def save_all_bundles(results, preprocessor, task_type, target_column,
                     feature_columns=None):
    """Save every trained model as its own .pkl. Returns {display_name: path}."""
    paths = {}
    for r in results:
        safe_name = (r["name"].replace(" ", "_").replace("(", "").replace(")", ""))
        bundle = {
            "model": r["fitted_model"],
            "model_name": r["name"],
            "source": r["source"],
            "task_type": task_type,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "preprocessor": preprocessor,
            "metrics_test": r.get("test_metrics", {}),
            "cv_mean": r["cv_mean"],
        }
        path = Config.MODEL_DIR / f"{safe_name}.pkl"
        joblib.dump(bundle, path)
        paths[r["name"]] = str(path)
        logger.info("Saved bundle → %s", path)
    return paths