import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

from config import Config
logger = logging.getLogger(__name__)

# =============================================================================
# Custom transformer
# =============================================================================
class OutlierHandler(BaseEstimator, TransformerMixin):
    """Clips each numeric column to [Q1 - k·IQR, Q3 + k·IQR], fitted on train only."""
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_: List[float] = []
        self.upper_bounds_: List[float] = []

    def fit(self, X, y=None):
        self.lower_bounds_ = []
        self.upper_bounds_ = []
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        for i in range(X_arr.shape[1]):
            col = X_arr[:, i]
            col = col[~pd.isna(col)]
            if len(col) == 0:
                self.lower_bounds_.append(-np.inf)
                self.upper_bounds_.append(np.inf)
                continue
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            self.lower_bounds_.append(q1 - self.factor * iqr)
            self.upper_bounds_.append(q3 + self.factor * iqr)
        return self

    def transform(self, X):
        X_arr = X.values.copy() if isinstance(X, pd.DataFrame) else np.asarray(X).copy()
        for i in range(X_arr.shape[1]):
            X_arr[:, i] = np.clip(X_arr[:, i], self.lower_bounds_[i], self.upper_bounds_[i])
        return X_arr


# =============================================================================
# Preprocessing
# =============================================================================
def build_preprocessing(X_train, X_val, X_test):
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    logger.info("Numeric features (%d): %s", len(numeric_features), numeric_features[:10])
    logger.info("Categorical features (%d): %s", len(categorical_features), categorical_features[:10])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier", OutlierHandler(factor=1.5)),
        ("scaler", RobustScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    selector = VarianceThreshold(threshold=0.0)
    X_train_p = selector.fit_transform(X_train_p)
    X_val_p = selector.transform(X_val_p)
    X_test_p = selector.transform(X_test_p)

    logger.info("Preprocessed shapes — Train: %s, Val: %s, Test: %s",
                X_train_p.shape, X_val_p.shape, X_test_p.shape)

    full_pipeline = Pipeline([("preprocessor", preprocessor), ("selector", selector)])
    return X_train_p, X_val_p, X_test_p, full_pipeline
