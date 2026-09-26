"""
the model collection (estimators + search spaces per task type).
"""
import logging
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)

from config import Config

logger = logging.getLogger(__name__)

# Optional XGBoost — degrades gracefully if libomp is missing (macOS)
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception as e:
    logger.warning("XGBoost unavailable (%s) — skipping.", type(e).__name__)
    HAS_XGB = False


def get_model_moka(problem_type: str) -> Dict[str, Tuple[Any, Dict]]:
    """Return {name: (estimator, search_space)} for the given task type."""
    if problem_type == "classification":
        moka = {
            "LogisticRegression": (
                LogisticRegression(max_iter=1000, random_state=Config.RANDOM_STATE),
                Config.CLASSIFICATION_SEARCH_SPACES["LogisticRegression"],
            ),
            "RandomForestClassifier": (
                RandomForestClassifier(
                    random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS,
                ),
                Config.CLASSIFICATION_SEARCH_SPACES["RandomForestClassifier"],
            ),
            "GradientBoostingClassifier": (
                GradientBoostingClassifier(random_state=Config.RANDOM_STATE),
                Config.CLASSIFICATION_SEARCH_SPACES["GradientBoostingClassifier"],
            ),
        }
        if HAS_XGB:
            moka["XGBClassifier"] = (
                XGBClassifier(
                    random_state=Config.RANDOM_STATE, eval_metric="logloss",
                ),
                Config.CLASSIFICATION_SEARCH_SPACES["XGBClassifier"],
            )
        return moka

    if problem_type == "regression":
        moka = {
            "Ridge": (
                Ridge(random_state=Config.RANDOM_STATE),
                Config.REGRESSION_SEARCH_SPACES["Ridge"],
            ),
            "RandomForestRegressor": (
                RandomForestRegressor(
                    random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS,
                ),
                Config.REGRESSION_SEARCH_SPACES["RandomForestRegressor"],
            ),
            "GradientBoostingRegressor": (
                GradientBoostingRegressor(random_state=Config.RANDOM_STATE),
                Config.REGRESSION_SEARCH_SPACES["GradientBoostingRegressor"],
            ),
        }
        if HAS_XGB:
            moka["XGBRegressor"] = (
                XGBRegressor(random_state=Config.RANDOM_STATE),
                Config.REGRESSION_SEARCH_SPACES["XGBRegressor"],
            )
        return moka

    raise ValueError(f"Unknown problem_type: {problem_type}")


def list_available_models(problem_type: str):
    return list(get_model_moka(problem_type).keys())