"""
Unit Tests for Recipe Optimization and Waste Calculation
Author: Khushi Yadav
"""

import pytest
from src.config import TOTAL_REGISTERED_STUDENTS
from src.optimizer import scale_recipe_quantities


def test_scale_recipe_quantities_lunch():
    result = scale_recipe_quantities("Lunch", 600, safety_buffer_pct=0.05)

    assert result["predicted_attendance"] == 600
    assert result["safety_buffer_pct"] == 5.0
    # 600 * 1.05 = 630
    assert result["buffered_target_headcount"] == 630
    assert result["total_composite_food_kg"] > 0
    assert "Rice (dry weight equivalent) (kg)" in result["ingredients"]
    assert "Chapatis / Rotis (pieces)" in result["ingredients"]
    assert result["ingredients"]["Rice (dry weight equivalent) (kg)"] == round(630 * 0.14, 1)


def test_scale_recipe_buffer_cap():
    # If prediction is 980 and buffer is 10%, it should not exceed TOTAL_REGISTERED_STUDENTS
    result = scale_recipe_quantities("Dinner", 980, safety_buffer_pct=0.10)
    assert result["buffered_target_headcount"] <= TOTAL_REGISTERED_STUDENTS
