"""
Smart Campus Mess Food Waste Predictor
A Machine Learning-driven attendance forecasting and dynamic recipe scaling system
designed to minimize food wastage in college hostel messes.
Author: Khushi Yadav
"""

from setuptools import setup, find_packages

setup(
    name="smart-campus-mess-food-waste-predictor",
    version="1.0.0",
    description="Machine Learning attendance forecasting & dynamic recipe scaling to prevent campus food waste",
    author="Khushi Yadav",
    author_email="ky19tech@gmail.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "joblib>=1.3.0",
        "streamlit>=1.30.0",
        "plotly>=5.18.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
