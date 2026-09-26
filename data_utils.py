# Data loading, X/y split, task-type detection, splitting.


import logging
from typing import Optional, Tuple

import pandas as pd

from sklearn.model_selection import train_test_split

from config import Config

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Load & store
# -----------------------------------------------------------------------------
def load_and_store_data(file) -> Tuple[Optional[pd.DataFrame], str]:

    if file is None:
        return None, "No file uploaded."

    try:
        path = file.name if hasattr(file, "name") else str(file)
        df = pd.read_csv(path)
    except Exception as e:
        logger.exception("CSV load failed")
        return None, f"Upload failed: {e}"

    if df.empty:
        return None, "Uploaded CSV is empty."

    logger.info("Loaded %s — shape=%s", path, df.shape)
    return df, f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns"


# -----------------------------------------------------------------------------
# X / y preparation
# -----------------------------------------------------------------------------
def prepare_xy(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into (X, y) and drop rows with a missing target.
    Raises ValueError on missing target or empty result.
    """
    if df is None:
        raise ValueError("No dataset loaded.")
    if target_column not in df.columns:
        raise ValueError(
            f"Target '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    y = df[target_column]
    X = df.drop(columns=[target_column])

    mask = ~y.isnull()
    dropped = int((~mask).sum())
    if dropped:
        logger.info("Dropping %d rows with missing target.", dropped)

    X, y = X[mask], y[mask]
    if len(y) == 0:
        raise ValueError("No rows remain after dropping missing target values.")

    logger.info("Prepared X=%s, y=%s", X.shape, y.shape)
    return X, y


# -----------------------------------------------------------------------------
# Task-type detection
# -----------------------------------------------------------------------------
def detect_problem_type(y: pd.Series) -> str:
    """
    - object / bool / category dtype          -> classification
    - numeric with <= 20 unique values        -> classification
    - numeric with >50 unique values      -> regression
    - otherwise numeric                       -> regression
    """
    y_clean = y.dropna()
    n_unique = y_clean.nunique()
    n_total = max(len(y_clean), 1)

    if pd.api.types.is_object_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    if isinstance(y.dtype, pd.CategoricalDtype):
        return "classification"

    if n_unique > 50 and (n_unique / n_total) > 0.05:
        return "regression"
    if n_unique <= 20:
        return "classification"
    return "regression"


# -----------------------------------------------------------------------------
# Splitting
# -----------------------------------------------------------------------------
def split_data(X, y):
    """60/20/20 train/val/test (deterministic via Config.RANDOM_STATE)."""
    X_full, X_test, y_full, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=Config.VAL_SIZE, random_state=Config.RANDOM_STATE,
    )
    logger.info("Split OK — Train: %s, Val: %s, Test: %s",
                X_train.shape, X_val.shape, X_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def get_shape(df: Optional[pd.DataFrame]) -> str:
    if df is None:
        return "No dataset"
    return f"{df.shape[0]} rows × {df.shape[1]} columns"


def find_missing_data(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=["Column", "Missing Count"])
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return pd.DataFrame(columns=["Column", "Missing Count"])
    return missing.reset_index().rename(columns={"index": "Column", 0: "Missing Count"})