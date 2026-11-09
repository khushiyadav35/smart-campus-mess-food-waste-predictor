"""
Campus Mess System Configuration
Defines campus parameters, meal definitions, recipe portions, and sustainability benchmarks.
Author: Khushi Yadav
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DATA_PATH = DATA_DIR / "mess_attendance_raw.csv"
PROCESSED_DATA_PATH = DATA_DIR / "mess_attendance_processed.csv"
MODEL_ARTIFACT_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
AUDIT_LOG_PATH = DATA_DIR / "mess_daily_audit_log.csv"

# Campus Demographics
TOTAL_REGISTERED_STUDENTS = 1000

# Traditional Mess Management Policy (Fixed Baseline)
# Most Indian college messes cook for a fixed 82-88% of registered hostellers
BASELINE_COOKING_RATIO = 0.85  # Cooks for 850 students every meal

# Meals definition
MEAL_TYPES = ["Breakfast", "Lunch", "Snacks", "Dinner"]

# Average base turnout by meal type under normal conditions
BASE_ATTENDANCE_RATIOS = {
    "Breakfast": 0.65,  # Students often skip breakfast (early lectures/sleeping in)
    "Lunch": 0.82,      # High weekday turnout between classes
    "Snacks": 0.55,     # Optional evening tea/refreshment
    "Dinner": 0.78,     # Stable weekday dinner turnout
}

# Standard Per-Student Recipe Portions (in kg or units)
# Based on National Institute of Nutrition (NIN) dietary guidelines for young adults
STANDARD_PORTIONS = {
    "Breakfast": {
        "Main Item (Idli/Dosa/Paratha/Poha) (kg)": 0.22,
        "Side / Chutney / Sambhar (kg)": 0.15,
        "Tea / Coffee / Milk (L)": 0.20,
    },
    "Lunch": {
        "Rice (dry weight equivalent) (kg)": 0.14,
        "Dal / Lentil Curry (kg)": 0.15,
        "Seasonal Sabzi / Paneer / Egg (kg)": 0.16,
        "Chapatis / Rotis (pieces)": 3.2,
        "Curd / Raita (kg)": 0.10,
    },
    "Snacks": {
        "Snack Item (Samosa/Pakora/Sandwich) (kg)": 0.15,
        "Tea / Coffee (L)": 0.18,
    },
    "Dinner": {
        "Rice / Pulao / Biryani (kg)": 0.15,
        "Dal / Kadhi / Gravy (kg)": 0.14,
        "Special Sabzi / Paneer / Chicken (kg)": 0.18,
        "Chapatis / Naan / Poori (pieces)": 3.4,
        "Dessert / Sweet (Gulab Jamun/Kheer) (kg)": 0.08,
    },
}

# Composite cooked food weight per student per meal (kg)
MEAL_COMPOSITE_WEIGHT_KG = {
    "Breakfast": 0.45,
    "Lunch": 0.65,
    "Snacks": 0.28,
    "Dinner": 0.70,
}

# Campus Menu Item Catalog with Popularity Index (1.0 to 10.0)
MENU_CATALOG = {
    "Breakfast": [
        {"name": "Aloo Paratha with Curd & Pickle", "popularity": 9.2, "category": "North Indian", "is_special": True},
        {"name": "Masala Dosa with Sambhar & Coconut Chutney", "popularity": 9.0, "category": "South Indian", "is_special": True},
        {"name": "Poha with Sev, Sprouts & Lemon", "popularity": 6.8, "category": "Light Breakfast", "is_special": False},
        {"name": "Poori Bhaji with Sooji Halwa", "popularity": 8.8, "category": "North Indian", "is_special": True},
        {"name": "Idli Vada with Sambhar & Chutney", "popularity": 7.2, "category": "South Indian", "is_special": False},
        {"name": "Upma with Coconut Chutney", "popularity": 4.5, "category": "Light Breakfast", "is_special": False},
        {"name": "Bread Omelette / Veg Sandwich", "popularity": 7.5, "category": "Continental", "is_special": False},
    ],
    "Lunch": [
        {"name": "Rajma Chawal + Mixed Veg + Boondi Raita + Rotis", "popularity": 9.3, "category": "North Indian", "is_special": True},
        {"name": "Chole Bhature / Poori + Mint Raita + Salad", "popularity": 9.5, "category": "Special", "is_special": True},
        {"name": "Kadhi Pakoda + Jeera Rice + Aloo Gobhi + Rotis", "popularity": 7.8, "category": "North Indian", "is_special": False},
        {"name": "Dal Tadka + Steamed Rice + Bhindi Masala + Rotis", "popularity": 7.0, "category": "Homestyle", "is_special": False},
        {"name": "South Indian Thali (Sambhar, Rasam, Poriyal, Curd)", "popularity": 6.5, "category": "South Indian", "is_special": False},
        {"name": "Lauki / Tinda Sabzi + Moong Dal + Rice + Rotis", "popularity": 3.8, "category": "Homestyle", "is_special": False},
        {"name": "Paneer Butter Masala + Peas Pulao + Dal Makhani + Rotis", "popularity": 9.8, "category": "North Indian", "is_special": True},
    ],
    "Snacks": [
        {"name": "Samosa with Mint & Tamarind Chutney + Masala Chai", "popularity": 9.1, "category": "Street Food", "is_special": True},
        {"name": "Pav Bhaji + Hot Chai", "popularity": 9.4, "category": "Street Food", "is_special": True},
        {"name": "Bread Pakora + Cutting Chai", "popularity": 8.2, "category": "Street Food", "is_special": False},
        {"name": "Vegetable Cutlet with Tomato Sauce + Tea", "popularity": 6.4, "category": "Snacks", "is_special": False},
        {"name": "Biscuits & Plain Tea", "popularity": 3.5, "category": "Simple", "is_special": False},
    ],
    "Dinner": [
        {"name": "Shahi Paneer / Chicken Curry + Butter Naan + Dal Makhani + Gulab Jamun", "popularity": 9.8, "category": "Feast / Weekend Special", "is_special": True},
        {"name": "Veg Biryani / Hyderabadi Chicken Biryani + Mirchi Ka Salan + Raita", "popularity": 9.7, "category": "Special", "is_special": True},
        {"name": "Dal Fry + Jeera Rice + Aloo Beans + Phulka + Kheer", "popularity": 6.6, "category": "Regular", "is_special": False},
        {"name": "Egg Curry / Malai Kofta + Steamed Rice + Phulka", "popularity": 8.4, "category": "Regular", "is_special": False},
        {"name": "Khichdi + Papad + Pickle + Curd (Detox Light Dinner)", "popularity": 4.1, "category": "Light Dinner", "is_special": False},
        {"name": "Matar Paneer + Tandoori Roti + Dal Tadka + Rice", "popularity": 8.9, "category": "North Indian", "is_special": True},
    ],
}

# Financial and Environmental Impact Constants
AVG_INGREDIENT_COST_PER_KG = 72.50   # In Indian Rupees (INR) per kg cooked equivalent
CO2E_KG_PER_KG_FOOD_WASTE = 2.50     # Life-cycle GHG footprint (methane in landfill + farming footprint)
DEFAULT_SAFETY_BUFFER = 0.04         # 4% safety margin above model prediction
