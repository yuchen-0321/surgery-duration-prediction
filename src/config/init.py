"""Configuration module"""

from .settings import DATA_CONFIG, FEATURE_CONFIG, MODEL_CONFIG, CV_CONFIG
from .departments import DEPARTMENT_CONFIGS, DepartmentConfig

__all__ = [
    "DATA_CONFIG",
    "FEATURE_CONFIG", 
    "MODEL_CONFIG",
    "CV_CONFIG",
    "DEPARTMENT_CONFIGS",
    "DepartmentConfig"
]
