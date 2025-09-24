"""部門特定設定"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DepartmentConfig:
    """部門設定類別"""
    name: str
    display_name: str
    best_params: Dict[str, Any]

# 各部門最佳參數（基於你的實驗結果）
DEPARTMENT_CONFIGS = {
    'ENT': DepartmentConfig(
        name='ENT',
        display_name='耳鼻喉科',
        best_params={
            'n_estimators': 300,
            'max_depth': None,
            'min_samples_split': 10,
            'min_samples_leaf': 1,
            'max_features': 1.0,
            'bootstrap': True
        }
    ),
    'GS': DepartmentConfig(
        name='GS',
        display_name='一般外科',
        best_params={
            'n_estimators': 500,
            'max_depth': 20,
            'min_samples_split': 15,
            'min_samples_leaf': 1,
            'max_features': 'log2',
            'bootstrap': True
        }
    ),
    'GU': DepartmentConfig(
        name='GU',
        display_name='泌尿外科',
        best_params={
            'n_estimators': 288,
            'max_depth': None,
            'min_samples_split': 17,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True
        }
    ),
    'OPH': DepartmentConfig(
        name='OPH',
        display_name='眼科',
        best_params={
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 12,
            'min_samples_leaf': 3,
            'max_features': 'sqrt',
            'bootstrap': True
        }
    ),
    'ORTH': DepartmentConfig(
        name='ORTH',
        display_name='骨科',
        best_params={
            'n_estimators': 350,
            'max_depth': None,
            'min_samples_split': 20,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True
        }
    )
}
