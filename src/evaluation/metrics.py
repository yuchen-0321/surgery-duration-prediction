"""模型評估指標"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型評估器"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """計算評估指標"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': ModelEvaluator._calculate_mape(y_true, y_pred)
        }
    
    @staticmethod
    def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """計算平均絕對百分比誤差"""
        # 避免除零錯誤
        mask = y_true != 0
        if not np.any(mask):
            return float('inf')
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """比較多個模型的效能"""
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.round(3)
        
        # 排序（R² 越高越好，其他指標越低越好）
        comparison_df['rank'] = (
            comparison_df['r2'].rank(ascending=False) +
            comparison_df['mse'].rank(ascending=True) +
            comparison_df['mae'].rank(ascending=True)
        )
        
        return comparison_df.sort_values('rank')
    
    @staticmethod
    def generate_evaluation_report(department: str, metrics: Dict[str, float]) -> str:
        """生成評估報告"""
        report = f"""
=== {department} 模型評估報告 ===

回歸指標:
- 均方誤差 (MSE): {metrics['mse']:.2f}
- 均方根誤差 (RMSE): {metrics['rmse']:.2f}
- 平均絕對誤差 (MAE): {metrics['mae']:.2f}
- 決定係數 (R²): {metrics['r2']:.3f}
- 平均絕對百分比誤差 (MAPE): {metrics['mape']:.2f}%

模型表現:
"""
        
        # 根據 R² 評估模型表現
        r2_score = metrics['r2']
        if r2_score >= 0.8:
            performance = "優秀"
        elif r2_score >= 0.7:
            performance = "良好"
        elif r2_score >= 0.6:
            performance = "一般"
        else:
            performance = "需改善"
            
        report += f"- 整體表現: {performance} (R² = {r2_score:.3f})"
        
        return report
