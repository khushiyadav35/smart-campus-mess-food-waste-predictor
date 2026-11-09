"""
Daily Mess Audit & Actual Waste Logger Service
Enables hostel mess supervisors and catering staff to log real-time end-of-meal actuals:
- Actual headcount served
- Actual measured leftover food buckets (kg)
- Auto-computes prediction deviation and logs feedback for continual learning.
Author: Khushi Yadav
"""

import os
import datetime
import pandas as pd
from typing import Dict, Any, List

from src.config import AUDIT_LOG_PATH, DATA_DIR


def init_audit_log():
    """Initializes audit CSV if not present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(AUDIT_LOG_PATH):
        df_init = pd.DataFrame(
            columns=[
                "timestamp",
                "date",
                "meal_type",
                "menu_name",
                "predicted_headcount",
                "cooked_headcount",
                "actual_headcount",
                "headcount_error",
                "actual_waste_kg",
                "waste_notes",
            ]
        )
        df_init.to_csv(AUDIT_LOG_PATH, index=False)


def record_mess_audit_entry(
    date_str: str,
    meal_type: str,
    menu_name: str,
    predicted_headcount: int,
    cooked_headcount: int,
    actual_headcount: int,
    actual_waste_kg: float,
    waste_notes: str = "",
) -> Dict[str, Any]:
    """Records an operational audit entry to the tracking CSV."""
    init_audit_log()

    error = actual_headcount - predicted_headcount
    now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = {
        "timestamp": now_iso,
        "date": date_str,
        "meal_type": meal_type,
        "menu_name": menu_name,
        "predicted_headcount": predicted_headcount,
        "cooked_headcount": cooked_headcount,
        "actual_headcount": actual_headcount,
        "headcount_error": error,
        "actual_waste_kg": actual_waste_kg,
        "waste_notes": waste_notes,
    }

    df_existing = pd.read_csv(AUDIT_LOG_PATH)
    df_updated = pd.concat([df_existing, pd.DataFrame([entry])], ignore_index=True)
    df_updated.to_csv(AUDIT_LOG_PATH, index=False)

    return entry


def get_recent_audit_logs(limit: int = 15) -> pd.DataFrame:
    """Retrieves recent audit entries."""
    init_audit_log()
    df = pd.read_csv(AUDIT_LOG_PATH)
    if df.empty:
        # Prepopulate with a few initial sample audit records so UI is immediately rich
        sample_records = [
            {
                "timestamp": "2026-11-07 14:15:00",
                "date": "2026-11-07",
                "meal_type": "Lunch",
                "menu_name": "Rajma Chawal + Mixed Veg + Boondi Raita",
                "predicted_headcount": 780,
                "cooked_headcount": 810,
                "actual_headcount": 792,
                "headcount_error": 12,
                "actual_waste_kg": 11.7,
                "waste_notes": "Slight excess curd, zero shortage reported.",
            },
            {
                "timestamp": "2026-11-07 21:40:00",
                "date": "2026-11-07",
                "meal_type": "Dinner",
                "menu_name": "Paneer Butter Masala + Naan + Kheer",
                "predicted_headcount": 840,
                "cooked_headcount": 875,
                "actual_headcount": 855,
                "headcount_error": 15,
                "actual_waste_kg": 14.0,
                "waste_notes": "Popular Friday dinner, smooth service.",
            },
            {
                "timestamp": "2026-11-08 14:20:00",
                "date": "2026-11-08",
                "meal_type": "Lunch",
                "menu_name": "Chole Bhature / Poori + Mint Raita",
                "predicted_headcount": 620,
                "cooked_headcount": 645,
                "actual_headcount": 632,
                "headcount_error": 12,
                "actual_waste_kg": 8.4,
                "waste_notes": "Saturday lunch, minimal leftovers.",
            },
        ]
        df_sample = pd.DataFrame(sample_records)
        df_sample.to_csv(AUDIT_LOG_PATH, index=False)
        return df_sample

    return df.tail(limit).iloc[::-1]  # Return newest first
