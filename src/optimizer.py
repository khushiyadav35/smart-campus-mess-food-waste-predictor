"""
Dynamic Recipe Scaling, Buffer Optimization, and Food Waste Reduction Analytics
Translates predicted student attendance into exact kitchen recipe batch quantities,
manages safety buffers to prevent shortages, and computes environmental & financial savings.
Author: Khushi Yadav
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from src.config import (
    TOTAL_REGISTERED_STUDENTS,
    BASELINE_COOKING_RATIO,
    STANDARD_PORTIONS,
    MEAL_COMPOSITE_WEIGHT_KG,
    AVG_INGREDIENT_COST_PER_KG,
    CO2E_KG_PER_KG_FOOD_WASTE,
    DEFAULT_SAFETY_BUFFER,
)


def scale_recipe_quantities(
    meal_type: str,
    predicted_attendance: int,
    safety_buffer_pct: float = DEFAULT_SAFETY_BUFFER,
) -> Dict[str, Any]:
    """
    Computes exact ingredient recipe cooking weights based on predicted attendance
    and mess manager safety buffer.
    """
    buffer_multiplier = 1.0 + safety_buffer_pct
    buffered_headcount = int(math.ceil(predicted_attendance * buffer_multiplier))
    buffered_headcount = min(buffered_headcount, TOTAL_REGISTERED_STUDENTS)

    portion_dict = STANDARD_PORTIONS.get(meal_type, STANDARD_PORTIONS["Lunch"])
    composite_unit_kg = MEAL_COMPOSITE_WEIGHT_KG.get(meal_type, 0.65)

    itemized_requirements = {}
    for item_name, unit_qty in portion_dict.items():
        total_req = round(buffered_headcount * unit_qty, 1)
        itemized_requirements[item_name] = total_req

    total_cooked_kg = round(buffered_headcount * composite_unit_kg, 1)

    return {
        "meal_type": meal_type,
        "predicted_attendance": predicted_attendance,
        "safety_buffer_pct": round(safety_buffer_pct * 100.0, 1),
        "buffered_target_headcount": buffered_headcount,
        "total_composite_food_kg": total_cooked_kg,
        "ingredients": itemized_requirements,
    }


def compute_waste_reduction_simulation(
    df_raw: pd.DataFrame,
    y_pred_attendance: np.ndarray,
    safety_buffer_pct: float = DEFAULT_SAFETY_BUFFER,
) -> Dict[str, Any]:
    """
    Simulates operational impact comparing traditional fixed cooking vs ML-driven smart cooking
    across test dataset:
    - Waste Reduction Percentage
    - Absolute Kilograms of Food Saved
    - Financial Savings (INR ₹)
    - Greenhouse Gas Emissions Offset (kg CO2e)
    - Shortage Frequency (Service Level)
    """
    test_slice = df_raw.iloc[-len(y_pred_attendance):].copy().reset_index(drop=True)

    test_slice["ml_pred_attendance"] = y_pred_attendance
    test_slice["ml_buffered_cook"] = np.ceil(
        test_slice["ml_pred_attendance"] * (1.0 + safety_buffer_pct)
    ).clip(upper=TOTAL_REGISTERED_STUDENTS)

    # Unit food weights
    portion_weights = test_slice["meal_type"].map(MEAL_COMPOSITE_WEIGHT_KG).values
    actual_attendees = test_slice["actual_attendance"].values
    baseline_cook = test_slice["baseline_cooked_meals"].values
    ml_cook = test_slice["ml_buffered_cook"].values

    # Food prepared in kg
    baseline_prepared_kg = baseline_cook * portion_weights
    ml_prepared_kg = ml_cook * portion_weights
    actual_consumed_kg = actual_attendees * portion_weights

    # Baseline Waste & Shortage
    baseline_waste = np.maximum(0, baseline_prepared_kg - actual_consumed_kg)
    baseline_shortage = np.maximum(0, actual_consumed_kg - baseline_prepared_kg)

    # ML Smart Mess Waste & Shortage
    ml_waste = np.maximum(0, ml_prepared_kg - actual_consumed_kg)
    ml_shortage = np.maximum(0, actual_consumed_kg - ml_prepared_kg)

    # Cumulative Metrics
    total_baseline_waste_kg = float(np.sum(baseline_waste))
    total_ml_waste_kg = float(np.sum(ml_waste))
    total_food_saved_kg = total_baseline_waste_kg - total_ml_waste_kg

    # Percentage Waste Reduction
    if total_baseline_waste_kg > 0:
        waste_reduction_pct = (total_food_saved_kg / total_baseline_waste_kg) * 100.0
    else:
        waste_reduction_pct = 0.0

    # Economics & Sustainability
    cost_saved_inr = total_food_saved_kg * AVG_INGREDIENT_COST_PER_KG
    co2e_prevented_kg = total_food_saved_kg * CO2E_KG_PER_KG_FOOD_WASTE

    # Reliability / Service Level (shortage events)
    total_meals = len(test_slice)
    baseline_shortage_events = int(np.sum(baseline_shortage > 0))
    ml_shortage_events = int(np.sum(ml_shortage > 0))

    baseline_service_level = round(100.0 * (1.0 - baseline_shortage_events / total_meals), 2)
    ml_service_level = round(100.0 * (1.0 - ml_shortage_events / total_meals), 2)

    # Daily Averages (Assuming 4 meals/day)
    num_days = max(1, total_meals / 4.0)
    daily_food_saved_kg = round(total_food_saved_kg / num_days, 1)
    daily_cost_saved_inr = round(cost_saved_inr / num_days, 1)

    return {
        "waste_reduction_pct": round(waste_reduction_pct, 1),
        "total_food_saved_kg": round(total_food_saved_kg, 1),
        "total_baseline_waste_kg": round(total_baseline_waste_kg, 1),
        "total_ml_waste_kg": round(total_ml_waste_kg, 1),
        "cost_saved_inr": round(cost_saved_inr, 2),
        "co2e_prevented_kg": round(co2e_prevented_kg, 1),
        "co2e_prevented_metric_tonnes": round(co2e_prevented_kg / 1000.0, 2),
        "daily_food_saved_kg": daily_food_saved_kg,
        "daily_cost_saved_inr": daily_cost_saved_inr,
        "evaluated_meals_count": total_meals,
        "evaluated_days_count": round(num_days, 0),
        "baseline_service_level_pct": baseline_service_level,
        "ml_service_level_pct": ml_service_level,
    }


if __name__ == "__main__":
    # Test recipe scaling
    res = scale_recipe_quantities("Lunch", 650, safety_buffer_pct=0.04)
    print("Sample Scaled Recipe for 650 Students Lunch:")
    print(res)
