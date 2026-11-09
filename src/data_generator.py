"""
Synthetic Campus Mess Attendance Data Generator
Simulates realistic multi-semester campus mess operational data reflecting:
- Academic calendars (Odd/Even semesters, Mid-term/End-term exams, Tech fests, Vacations)
- Temporal patterns (Day-of-week attrition, Friday dinner exodus, Sunday brunch)
- Indian public holidays and long-weekend exodus effects
- Menu popularity dynamics (Biryani/Paneer surges vs Lauki/Khichdi drops)
- Local weather factors (Monsoon rainfall spikes, extreme summer heat)
Author: Khushi Yadav
"""

import os
import random
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from src.config import (
    TOTAL_REGISTERED_STUDENTS,
    BASELINE_COOKING_RATIO,
    MEAL_TYPES,
    BASE_ATTENDANCE_RATIOS,
    MENU_CATALOG,
    MEAL_COMPOSITE_WEIGHT_KG,
    AVG_INGREDIENT_COST_PER_KG,
    RAW_DATA_PATH,
    DATA_DIR,
)

# Set seeds for complete scientific reproducibility
np.random.seed(42)
random.seed(42)

# Gazetted Holidays / Festival calendar (Month, Day, Name)
INDIAN_HOLIDAYS = [
    (1, 26, "Republic Day"),
    (3, 8, "Maha Shivratri"),
    (3, 25, "Holi"),
    (4, 11, "Eid-ul-Fitr"),
    (4, 14, "Ambedkar Jayanti"),
    (8, 15, "Independence Day"),
    (8, 26, "Janmashtami"),
    (10, 2, "Gandhi Jayanti"),
    (10, 12, "Dussehra"),
    (10, 31, "Diwali"),
    (11, 15, "Guru Nanak Jayanti"),
    (12, 25, "Christmas"),
]


def is_gazetted_holiday(dt: datetime.date) -> tuple[bool, str]:
    """Check if date matches an Indian festival or gazetted holiday."""
    for m, d, name in INDIAN_HOLIDAYS:
        if dt.month == m and dt.day == d:
            return True, name
    return False, "None"


def get_academic_status(dt: datetime.date) -> tuple[str, bool]:
    """
    Simulates standard Indian engineering university academic calendar:
    - Odd Sem: July - Nov (Midsem: Late Sept, Endsem: Mid Nov)
    - Winter Break: Dec 1 - Dec 31
    - Even Sem: Jan - May (Midsem: Early March, Endsem: Early May)
    - Summer Break: June 1 - June 30
    - Annual College Fest: 3rd weekend of February
    """
    month = dt.month
    day = dt.day

    # Summer / Winter Vacations
    if month == 6:
        return "Summer Vacation", False
    if month == 12:
        return "Winter Vacation", False

    # Exam Periods
    # Mid-sem odd: Sep 20 - Sep 28
    if month == 9 and 20 <= day <= 28:
        return "Mid-Term Exams", True
    # End-sem odd: Nov 15 - Nov 30
    if month == 11 and 15 <= day <= 30:
        return "End-Term Exams", True
    # Mid-sem even: Mar 10 - Mar 18
    if month == 3 and 10 <= day <= 18:
        return "Mid-Term Exams", True
    # End-sem even: May 5 - May 20
    if month == 5 and 5 <= day <= 20:
        return "End-Term Exams", True

    # Preparatory Leave (week before end-sems)
    if (month == 11 and 8 <= day <= 14) or (month == 4 and 27 <= day <= 30):
        return "Preparatory Leave", False

    # College Cultural Fest (Mid-February)
    if month == 2 and 18 <= day <= 21:
        return "College Cultural Fest", False

    return "Regular Academic Session", False


def simulate_weather(dt: datetime.date) -> tuple[float, float, bool]:
    """
    Simulates realistic north-central Indian weather conditions:
    - Summer (Apr-Jun): 34°C - 44°C, low rain
    - Monsoon (Jul-Sep): 26°C - 33°C, frequent heavy rain
    - Post-monsoon / Autumn (Oct-Nov): 20°C - 29°C
    - Winter (Dec-Feb): 9°C - 22°C
    """
    month = dt.month

    if month in [12, 1, 2]:
        temp = np.random.normal(16.5, 3.2)
        rainfall_prob = 0.08
    elif month in [3, 4, 5]:
        temp = np.random.normal(36.0, 3.5)
        rainfall_prob = 0.05
    elif month in [6, 7, 8, 9]:
        temp = np.random.normal(29.5, 2.5)
        rainfall_prob = 0.52  # Active monsoon
    else:
        temp = np.random.normal(24.0, 2.8)
        rainfall_prob = 0.12

    temp = round(float(np.clip(temp, 7.0, 46.0)), 1)

    is_rainy = random.random() < rainfall_prob
    if is_rainy:
        rainfall_mm = round(float(np.random.exponential(scale=18.0) + 2.0), 1)
    else:
        rainfall_mm = 0.0

    return temp, rainfall_mm, is_rainy


def generate_campus_dataset(
    start_date: datetime.date = datetime.date(2024, 7, 1),
    num_days: int = 730,
    total_students: int = TOTAL_REGISTERED_STUDENTS,
) -> pd.DataFrame:
    """
    Generates a full 2-year (730 days) campus mess dataset across Breakfast, Lunch, Snacks, and Dinner.
    Returns 2,920 meal observations.
    """
    records: List[Dict[str, Any]] = []
    baseline_cook_count = int(total_students * BASELINE_COOKING_RATIO)

    current_date = start_date

    # Track rolling attendance for realism
    prev_meal_attendance = int(total_students * 0.75)

    for day_idx in range(num_days):
        dt = current_date + datetime.timedelta(days=day_idx)
        dow = dt.weekday()  # 0=Monday, 6=Sunday
        day_name = dt.strftime("%A")
        is_weekend = dow in [5, 6]

        # Check calendar states
        academic_status, is_exam = get_academic_status(dt)
        is_holiday, holiday_name = is_gazetted_holiday(dt)

        # Long weekend check (Holiday on Friday or Monday)
        is_long_weekend = False
        if is_holiday and (dow in [0, 4]):
            is_long_weekend = True
        elif is_weekend:
            # Check adjacent Friday or Monday
            prev_fri = dt - datetime.timedelta(days=(dow - 4))
            next_mon = dt + datetime.timedelta(days=(7 - dow))
            if is_gazetted_holiday(prev_fri)[0] or is_gazetted_holiday(next_mon)[0]:
                is_long_weekend = True

        # Weather
        temperature_c, rainfall_mm, is_rainy = simulate_weather(dt)

        # Iterate 4 meals per day
        for meal in MEAL_TYPES:
            base_ratio = BASE_ATTENDANCE_RATIOS[meal]
            portion_weight_kg = MEAL_COMPOSITE_WEIGHT_KG[meal]

            # Pick a realistic menu from catalog
            available_menus = MENU_CATALOG[meal]
            menu_item = random.choice(available_menus)
            menu_name = menu_item["name"]
            menu_popularity = menu_item["popularity"]
            is_special = menu_item["is_special"]
            menu_category = menu_item["category"]

            # Multiplier calculation based on operational factors:
            factor = 1.0

            # 1. Vacation effect
            if academic_status in ["Summer Vacation", "Winter Vacation"]:
                factor *= 0.22  # Only ~20% students stay in hostel

            # 2. Fest effect
            elif academic_status == "College Cultural Fest":
                if meal in ["Lunch", "Dinner", "Snacks"]:
                    factor *= 0.58  # Students eat at festival food stalls

            # 3. Exam period effect
            elif is_exam:
                if meal == "Breakfast":
                    factor *= 0.78  # Sleep deprivation, rush to morning exam
                elif meal == "Lunch":
                    factor *= 0.98  # Quick mess lunch between sessions
                elif meal == "Snacks":
                    factor *= 1.25  # Study breaks / tea demand
                elif meal == "Dinner":
                    factor *= 1.12  # Studying inside hostel campus

            elif academic_status == "Preparatory Leave":
                if meal in ["Lunch", "Dinner"]:
                    factor *= 0.88  # Group study / library / outside ordering

            # 4. Long Weekend & Weekend Drop
            if is_long_weekend:
                factor *= 0.42  # Mass exodus to hometowns
            elif is_holiday:
                factor *= 0.60
            elif dow == 4 and meal == "Dinner":  # Friday dinner drop
                factor *= 0.68
            elif dow == 5:  # Saturday
                if meal == "Breakfast":
                    factor *= 0.62  # Sleeping late
                elif meal == "Lunch":
                    factor *= 0.74
                elif meal == "Dinner":
                    factor *= 0.60  # Going out with friends
            elif dow == 6:  # Sunday
                if meal == "Breakfast":
                    factor *= 0.55
                elif meal == "Lunch":
                    factor *= 0.90  # Sunday special feast
                elif meal == "Dinner":
                    factor *= 0.65

            # 5. Menu Popularity Impact (-25% for unpopular dishes to +20% for top favorites)
            # Popularity is calibrated on 1-10 scale (neutral is ~6.5)
            menu_delta = (menu_popularity - 6.5) * 0.038
            factor += menu_delta

            # 6. Rainy Weather Impact (+8% to +14% because stepping outside to cafes is prevented)
            if is_rainy:
                factor += 0.09 if rainfall_mm > 15 else 0.05

            # 7. Extreme Temperature Effect (Heatwave lunch reduction)
            if temperature_c > 41.0 and meal == "Lunch":
                factor -= 0.06

            # Compute actual attendance with natural student turnout stochasticity
            mean_turnout = total_students * base_ratio * factor
            noise = np.random.normal(0, total_students * 0.025)
            attendance = int(np.clip(mean_turnout + noise, 40, total_students - 15))

            # Fixed Baseline Mess Cooking (Status Quo)
            # In typical colleges, the mess manager always cooks for fixed capacity
            fixed_food_prepared_kg = round(baseline_cook_count * portion_weight_kg, 2)
            actual_food_consumed_kg = round(attendance * portion_weight_kg, 2)

            # Food Waste in Baseline Approach
            if fixed_food_prepared_kg > actual_food_consumed_kg:
                baseline_waste_kg = round(fixed_food_prepared_kg - actual_food_consumed_kg, 2)
                shortage_kg = 0.0
            else:
                baseline_waste_kg = 0.0
                shortage_kg = round(actual_food_consumed_kg - fixed_food_prepared_kg, 2)

            baseline_waste_cost_inr = round(baseline_waste_kg * AVG_INGREDIENT_COST_PER_KG, 2)

            # Record
            record = {
                "date": dt.isoformat(),
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "day_of_week": dow,
                "day_name": day_name,
                "is_weekend": int(is_weekend),
                "academic_status": academic_status,
                "is_exam": int(is_exam),
                "is_holiday": int(is_holiday),
                "is_long_weekend": int(is_long_weekend),
                "holiday_name": holiday_name,
                "meal_type": meal,
                "menu_name": menu_name,
                "menu_category": menu_category,
                "menu_popularity": menu_popularity,
                "is_special_menu": int(is_special),
                "temperature_c": temperature_c,
                "rainfall_mm": rainfall_mm,
                "is_rainy": int(is_rainy),
                "prev_meal_attendance": prev_meal_attendance,
                "registered_students": total_students,
                "actual_attendance": attendance,
                "baseline_cooked_meals": baseline_cook_count,
                "baseline_food_prepared_kg": fixed_food_prepared_kg,
                "actual_food_consumed_kg": actual_food_consumed_kg,
                "baseline_waste_kg": baseline_waste_kg,
                "baseline_waste_cost_inr": baseline_waste_cost_inr,
            }
            records.append(record)
            prev_meal_attendance = attendance

    df = pd.DataFrame(records)

    # Compute rolling 7-day attendance feature (lagged by 1 meal to avoid data leakage)
    df["rolling_7d_attendance"] = (
        df.groupby("meal_type")["actual_attendance"]
        .transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).mean())
        .bfill()
        .round(1)
    )

    return df


def main():
    """Generates the raw mess operational dataset and saves to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Simulating 2 academic years of campus mess attendance logs...")
    df = generate_campus_dataset()
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Dataset generated successfully with {len(df)} meal logs.")
    print(f"Saved to: {RAW_DATA_PATH}")
    print(f"Summary Statistics:\n{df[['actual_attendance', 'baseline_waste_kg']].describe()}")


if __name__ == "__main__":
    main()
