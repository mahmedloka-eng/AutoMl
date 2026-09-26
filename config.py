
# Global configuration

import os
from pathlib import Path


class Config:
    # --- Reproducibility ---
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2                  # of the train+val portion
    CV_FOLDS = 5
    N_JOBS = -1
    LOG_LEVEL = "INFO"

    # --- Directories ---
    ROOT = Path(__file__).resolve().parent
    MODEL_DIR = ROOT / "models"
    EXPERIMENT_DIR = ROOT / "experiments"

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    # --- Classification search spaces (multiclass-safe solvers only) ---
    CLASSIFICATION_SEARCH_SPACES = {
        "LogisticRegression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs", "saga", "newton-cg"],
        },
        "RandomForestClassifier": {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "GradientBoostingClassifier": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        },
        "XGBClassifier": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        },
    }

    # --- Regression search spaces ---
    REGRESSION_SEARCH_SPACES = {
        "Ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "RandomForestRegressor": {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "GradientBoostingRegressor": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        },
        "XGBRegressor": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        },
    }

    # --- Tuning ---
    N_ITER_SEARCH = 5               # RandomizedSearchCV iterations per model
    TUNE_EVERY_MODEL = True         # True → tune all baselines; False → top-K only
    TOP_K_TO_TUNE = 2               # only used if TUNE_EVERY_MODEL is False