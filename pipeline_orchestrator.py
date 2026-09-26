# Orchestrates the end-to-end auto_Ml pipeline: preprocess → train → tune → eval → save.

import logging
from typing import Any, Dict

import pandas as pd
from config import Config
from data_utils import prepare_xy, split_data, detect_problem_type
from model_moka import get_model_moka                     
from model_utils import (
    train_models, select_tuning_candidates, tune_models, retrain_tuned,
    pick_final_winner, evaluate_test, save_all_bundles, build_leaderboard,
    _safe_mlflow_log, HAS_MLFLOW,
)
from preprocessing import build_preprocessing

logger = logging.getLogger(__name__)


# =============================================================================
# Orchestrator

def run_full_pipeline(df: pd.DataFrame,
                      target: str,
                      task_type: str,
                      model_choice: str = "all",
                      do_tune: bool = True) -> Dict[str, Any]:
    """
    Runs the full auto_Ml pipeline and returns a dict with:
      - leaderboard, all_results, result_by_name, bundle_paths,
      - winner_name, winner_fig, split_msg, prep_msg, status
    Raises Exception on hard failure.
    """
    # ---- 1. Prepare X/y + split ----
    X, y = prepare_xy(df, target)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    split_msg = f"✅ Split: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"

    # ---- 2. Preprocess ----
    X_train_p, X_val_p, X_test_p, preprocessor = build_preprocessing(X_train, X_val, X_test)
    prep_msg = f"✅ Preprocessed: {X_train_p.shape[1]} features"

    # ---- 3. Baseline ----
    names = None if model_choice == "all" else [model_choice]
    _lb, baseline_results, _trained, base_msg = train_models(
        X_train_p, y_train, X_val_p, y_val, task_type, model_names=names,
    )
    if baseline_results is None:
        raise RuntimeError(base_msg)

    # ---- 4. Tune ----
    tuned_results = []
    if do_tune:
        top_k = None if Config.TUNE_EVERY_MODEL else Config.TOP_K_TO_TUNE
        candidates = select_tuning_candidates(baseline_results, top_k=top_k)
        logger.info("Tuning %d models: %s",
                    len(candidates), [c["name"] for c in candidates])
        tuning_info = tune_models(X_train_p, y_train, task_type, candidates)
        tuned_results = retrain_tuned(
            X_train_p, y_train, X_val_p, y_val, task_type, tuning_info
        )

    # ---- 5. Evaluate every model on test ----
    all_results = list(baseline_results) + list(tuned_results)
    for r in all_results:
        try:
            m, fig = evaluate_test(r["fitted_model"], X_test_p, y_test, task_type)
            r["test_metrics"] = m
            r["test_fig"] = fig
        except Exception as e:
            logger.warning("Test eval failed for %s: %s", r["name"], e)
            r["test_metrics"] = {}
            r["test_fig"] = None

    # ---- 5b. Log test metrics + preprocessor to MLflow ----
    if HAS_MLFLOW:
        for r in all_results:
            try:
                _safe_mlflow_log(
                    r["fitted_model"], r["name"],
                    r["train_metrics"], r["val_metrics"],
                    r["cv_mean"], r["cv_std"],
                    stage="testlog",
                    extra={
                        "task_type": task_type,
                        "target_column": target,
                        "source": r["source"],
                        "test_metrics": r.get("test_metrics", {}),
                        "preprocessor": preprocessor,
                    },
                )
            except Exception as e:
                logger.warning("Test-log failed for %s: %s", r["name"], e)

    # ---- 6. Save all bundles ----
    try:
        bundle_paths = save_all_bundles(
            all_results, preprocessor, task_type, target,
            feature_columns=list(X.columns),
        )
    except Exception as e:
        logger.exception("Saving bundles failed")
        bundle_paths = {}

    # ---- 7. Winner ----
    winner = pick_final_winner(baseline_results, tuned_results)
    winner_name = winner["name"]

    # ---- 8. Leaderboard ----
    leaderboard = build_leaderboard(all_results, task_type)

    status = (
        f"{split_msg}\n{prep_msg}\n\n"
        f"🏆 Best by CV: {winner_name} ({winner['source']})\n"
        f"CV mean: {winner['cv_mean']:.4f}\n"
        f"Saved {len(bundle_paths)} model bundles in models/\n"
        f"MLflow runs written to experiments/\n"
        f"Use the dropdowns below to view or download any model."
    )

    return {
        "leaderboard": leaderboard,
        "all_results": all_results,
        "result_by_name": {r["name"]: r for r in all_results},
        "bundle_paths": bundle_paths,
        "winner_name": winner_name,
        "winner_fig": next((r.get("test_fig") for r in all_results
                            if r["name"] == winner_name), None),
        "split_msg": split_msg,
        "prep_msg": prep_msg,
        "status": status,
    }


def build_moka_for_target(y: pd.Series):
    """Used by the UI to auto-fill the model dropdown."""
    task = detect_problem_type(y)
    return task, get_model_moka(task)