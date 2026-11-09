"""
Unit Tests for Model Artifacts and Evaluation Utilities
Author: Khushi Yadav
"""

import os
import joblib
import numpy as np
import pytest
from src.config import MODEL_ARTIFACT_PATH, PREPROCESSOR_PATH, TOTAL_REGISTERED_STUDENTS
from src.models import calculate_mape, evaluate_model


def test_metrics_calculation():
    y_true = np.array([500, 600, 700, 800])
    y_pred = np.array([510, 590, 710, 790])

    metrics = evaluate_model("TestModel", y_true, y_pred)
    assert metrics["MAE"] == 10.0
    assert metrics["RMSE"] == 10.0
    assert metrics["MAPE_percent"] < 3.0
    assert metrics["R2"] > 0.95


def test_saved_model_artifact_loading():
    assert os.path.exists(MODEL_ARTIFACT_PATH), "Model artifact should exist"
    assert os.path.exists(PREPROCESSOR_PATH), "Preprocessor artifact should exist"

    model = joblib.load(MODEL_ARTIFACT_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    assert hasattr(model, "predict")
    assert hasattr(preprocessor, "transform")
