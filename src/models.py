"""
Model Training, Benchmarking, and Evaluation Pipeline
Implements:
- Fixed-capacity baseline & 7-day moving average baseline
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor
Evaluates on out-of-sample temporal test split with MAE, RMSE, MAPE, and R2 metrics.
Saves the best-performing model, preprocessor, and metrics JSON.
Author: Khushi Yadav
"""

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    TOTAL_REGISTERED_STUDENTS,
    BASELINE_COOKING_RATIO,
    MODEL_ARTIFACT_PATH,
    PREPROCESSOR_PATH,
    METRICS_PATH,
    MODELS_DIR,
)
from src.preprocessor import prepare_training_data, CampusMessPreprocessor


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Percentage Error (MAPE %)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Filter out near-zero denominators
    mask = y_true > 10
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = calculate_mape(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_percent": round(mape, 2),
        "R2": round(r2, 4),
    }


def train_and_benchmark(
    test_ratio: float = 0.20,
) -> Tuple[Any, CampusMessPreprocessor, Dict[str, Any]]:
    """
    Executes model training pipeline across all candidate models and baselines.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Features
    X, y, preprocessor = prepare_training_data()

    # 2. Chronological Train-Test Split (Preserving real-world time-series order)
    split_idx = int(len(X) * (1.0 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\n--- Chronological Data Split ---")
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. Baselines
    # Baseline A: Traditional Fixed Cooking (85% of 1000 = 850)
    fixed_pred_test = np.full(shape=len(y_test), fill_value=int(TOTAL_REGISTERED_STUDENTS * BASELINE_COOKING_RATIO))
    metrics_fixed = evaluate_model("Fixed Capacity (Status Quo)", y_test.values, fixed_pred_test)

    # Baseline B: Rolling 7-Day Attendance feature directly
    rolling_pred_test = X_test["rolling_7d_attendance"].values
    metrics_rolling = evaluate_model("7-Day Moving Average", y_test.values, rolling_pred_test)

    # 4. Machine Learning Models
    models = {
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=160,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=180,
            learning_rate=0.06,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
        ),
    }

    all_metrics = {
        "Fixed Baseline (85%)": metrics_fixed,
        "7-Day Moving Average": metrics_rolling,
    }

    trained_instances = {}
    best_model_name = ""
    best_mae = float("inf")

    print("\n--- Training & Evaluating Candidate Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Realistic attendance bound clipping
        y_pred = np.clip(y_pred, 30, TOTAL_REGISTERED_STUDENTS)

        metrics = evaluate_model(name, y_test.values, y_pred)
        all_metrics[name] = metrics
        trained_instances[name] = model

        print(f"  -> MAE: {metrics['MAE']} students | RMSE: {metrics['RMSE']} | MAPE: {metrics['MAPE_percent']}% | R2: {metrics['R2']}")

        if metrics["MAE"] < best_mae:
            best_mae = metrics["MAE"]
            best_model_name = name

    print(f"\nWinning Model: {best_model_name} (MAE: {best_mae} students)")
    winning_model = trained_instances[best_model_name]

    # 5. Extract Feature Importance
    if hasattr(winning_model, "feature_importances_"):
        importances = winning_model.feature_importances_
        feature_importance_dict = {
            feat: round(float(imp), 4)
            for feat, imp in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)[:15]
        }
    else:
        feature_importance_dict = {}

    summary_payload = {
        "best_model_name": best_model_name,
        "benchmark_metrics": all_metrics,
        "top_feature_importance": feature_importance_dict,
        "test_samples": len(y_test),
        "training_samples": len(y_train),
    }

    # 6. Save Artifacts
    joblib.dump(winning_model, MODEL_ARTIFACT_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(summary_payload, f, indent=4)

    print(f"Model saved to: {MODEL_ARTIFACT_PATH}")
    print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")

    return winning_model, preprocessor, summary_payload


if __name__ == "__main__":
    train_and_benchmark()
