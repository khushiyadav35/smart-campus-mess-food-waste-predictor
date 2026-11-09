"""
Feature Engineering & Data Preprocessing Pipeline
Transforms raw campus mess operational data into model-ready numerical feature matrices:
- Cyclical sine/cosine temporal encodings for Day-of-Week and Month
- Categorical one-hot encodings for Meals, Academic Phases, and Menu Categories
- Weather and student behavioral indicators
- Lag attendance and rolling weekly moving averages
Author: Khushi Yadav
"""

import math
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    MEAL_TYPES,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
)

# Canonical feature list
NUMERICAL_FEATURES = [
    "sin_dow",
    "cos_dow",
    "sin_month",
    "cos_month",
    "is_weekend",
    "is_exam",
    "is_holiday",
    "is_long_weekend",
    "menu_popularity",
    "is_special_menu",
    "temperature_c",
    "rainfall_mm",
    "is_rainy",
    "prev_meal_attendance",
    "rolling_7d_attendance",
]

CATEGORICAL_FEATURES = [
    "meal_type",
    "academic_status",
    "menu_category",
]

TARGET_COLUMN = "actual_attendance"


class CampusMessPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compliant feature engineering transformer for campus mess attendance data.
    Provides consistent transformation between training batches and real-time single meal inference.
    """

    def __init__(self):
        self.feature_names_: List[str] = []
        self.meal_type_cats_: List[str] = []
        self.academic_cats_: List[str] = []
        self.category_cats_: List[str] = []
        self.default_prev_attendance_: float = 550.0
        self.default_rolling_attendance_: float = 580.0

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes cyclical, logical and interaction features."""
        data = df.copy()

        # Parse date if provided as string
        if "date" in data.columns and not pd.api.types.is_datetime64_any_dtype(data["date"]):
            data["date"] = pd.to_datetime(data["date"])
            data["day_of_week"] = data["date"].dt.dayofweek
            data["month"] = data["date"].dt.month

        # Cyclical Day-of-Week (period = 7)
        dow = data["day_of_week"].astype(float)
        data["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
        data["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

        # Cyclical Month (period = 12)
        month = data["month"].astype(float) - 1.0
        data["sin_month"] = np.sin(2 * np.pi * month / 12.0)
        data["cos_month"] = np.cos(2 * np.pi * month / 12.0)

        # Fill default rolling values if missing
        if "prev_meal_attendance" not in data.columns or data["prev_meal_attendance"].isna().any():
            data["prev_meal_attendance"] = data.get("prev_meal_attendance", pd.Series(self.default_prev_attendance_, index=data.index)).fillna(self.default_prev_attendance_)

        if "rolling_7d_attendance" not in data.columns or data["rolling_7d_attendance"].isna().any():
            data["rolling_7d_attendance"] = data.get("rolling_7d_attendance", pd.Series(self.default_rolling_attendance_, index=data.index)).fillna(self.default_rolling_attendance_)

        # Ensure binary columns are integer flags
        binary_cols = ["is_weekend", "is_exam", "is_holiday", "is_long_weekend", "is_special_menu", "is_rainy"]
        for col in binary_cols:
            if col in data.columns:
                data[col] = data[col].astype(int)
            else:
                data[col] = 0

        return data

    def fit(self, X: pd.DataFrame, y=None):
        """Learns categorical levels and reference statistics from training data."""
        data = self._engineer_features(X)

        self.meal_type_cats_ = sorted(list(data["meal_type"].unique()))
        self.academic_cats_ = sorted(list(data["academic_status"].unique()))
        self.category_cats_ = sorted(list(data["menu_category"].unique()))

        if "actual_attendance" in data.columns:
            self.default_prev_attendance_ = float(data["actual_attendance"].mean())
            self.default_rolling_attendance_ = float(data["actual_attendance"].mean())

        # Build final feature names list
        cat_encoded_cols = (
            [f"meal_{c}" for c in self.meal_type_cats_]
            + [f"acad_{c}" for c in self.academic_cats_]
            + [f"cat_{c}" for c in self.category_cats_]
        )
        self.feature_names_ = NUMERICAL_FEATURES + cat_encoded_cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame into the model feature space."""
        data = self._engineer_features(X)

        # One-hot encoding for categorical variables with fixed vocabulary
        for cat in self.meal_type_cats_:
            data[f"meal_{cat}"] = (data["meal_type"] == cat).astype(int)

        for cat in self.academic_cats_:
            data[f"acad_{cat}"] = (data["academic_status"] == cat).astype(int)

        for cat in self.category_cats_:
            data[f"cat_{cat}"] = (data["menu_category"] == cat).astype(int)

        # Return DataFrame aligned with feature_names_
        return data[self.feature_names_]

    def transform_single(self, input_dict: Dict[str, Any]) -> pd.DataFrame:
        """Transforms a single meal scenario dictionary (from UI) for real-time model inference."""
        df_single = pd.DataFrame([input_dict])
        return self.transform(df_single)


def prepare_training_data(
    raw_path: str = str(RAW_DATA_PATH),
    processed_path: str = str(PROCESSED_DATA_PATH),
) -> Tuple[pd.DataFrame, pd.Series, CampusMessPreprocessor]:
    """
    Loads raw CSV, fits preprocessor, transforms features, and writes processed dataset.
    """
    print(f"Loading raw mess dataset from: {raw_path}")
    raw_df = pd.read_csv(raw_path)

    preprocessor = CampusMessPreprocessor()
    preprocessor.fit(raw_df)

    X = preprocessor.transform(raw_df)
    y = raw_df[TARGET_COLUMN]

    # Save processed dataset for transparency and notebook inspection
    processed_df = X.copy()
    processed_df[TARGET_COLUMN] = y
    processed_df.to_csv(processed_path, index=False)
    print(f"Processed dataset saved to: {processed_path}")
    print(f"Engineered feature count: {X.shape[1]} features across {X.shape[0]} samples.")

    return X, y, preprocessor


if __name__ == "__main__":
    prepare_training_data()
