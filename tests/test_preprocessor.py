"""
Unit Tests for Campus Mess Feature Preprocessor
Author: Khushi Yadav
"""

import pytest
import numpy as np
import pandas as pd
from src.config import TOTAL_REGISTERED_STUDENTS
from src.preprocessor import CampusMessPreprocessor


@pytest.fixture
def sample_mess_data():
    return pd.DataFrame([
        {
            "date": "2026-11-09",
            "year": 2026,
            "month": 11,
            "day": 9,
            "day_of_week": 0,
            "day_name": "Monday",
            "is_weekend": 0,
            "academic_status": "Regular Academic Session",
            "is_exam": 0,
            "is_holiday": 0,
            "is_long_weekend": 0,
            "meal_type": "Lunch",
            "menu_name": "Rajma Chawal + Mixed Veg + Boondi Raita",
            "menu_category": "North Indian",
            "menu_popularity": 9.3,
            "is_special_menu": 1,
            "temperature_c": 26.5,
            "rainfall_mm": 0.0,
            "is_rainy": 0,
            "prev_meal_attendance": 650,
            "rolling_7d_attendance": 640.0,
            "actual_attendance": 780,
        },
        {
            "date": "2026-11-15",
            "year": 2026,
            "month": 11,
            "day": 15,
            "day_of_week": 6,
            "day_name": "Sunday",
            "is_weekend": 1,
            "academic_status": "End-Term Exams",
            "is_exam": 1,
            "is_holiday": 0,
            "is_long_weekend": 0,
            "meal_type": "Dinner",
            "menu_name": "Shahi Paneer + Butter Naan + Dal Makhani",
            "menu_category": "Special",
            "menu_popularity": 9.8,
            "is_special_menu": 1,
            "temperature_c": 21.0,
            "rainfall_mm": 15.0,
            "is_rainy": 1,
            "prev_meal_attendance": 720,
            "rolling_7d_attendance": 680.0,
            "actual_attendance": 840,
        },
    ])


def test_preprocessor_fit_transform(sample_mess_data):
    preprocessor = CampusMessPreprocessor()
    preprocessor.fit(sample_mess_data)
    X = preprocessor.transform(sample_mess_data)

    assert isinstance(X, pd.DataFrame)
    assert len(X) == 2
    assert "sin_dow" in X.columns
    assert "cos_dow" in X.columns
    assert "sin_month" in X.columns
    assert "cos_month" in X.columns

    # Verify cyclical encodings stay in valid [-1.0, 1.0] range
    assert (X["sin_dow"] >= -1.0).all() and (X["sin_dow"] <= 1.0).all()
    assert (X["cos_dow"] >= -1.0).all() and (X["cos_dow"] <= 1.0).all()


def test_preprocessor_single_inference(sample_mess_data):
    preprocessor = CampusMessPreprocessor()
    preprocessor.fit(sample_mess_data)

    single_dict = sample_mess_data.iloc[0].to_dict()
    X_single = preprocessor.transform_single(single_dict)

    assert len(X_single) == 1
    assert list(X_single.columns) == preprocessor.feature_names_
