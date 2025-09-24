"""學習曲線分析"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class LearningCurveAnalyzer:
    """學習曲線分析器"""
    
    def __init__(self, style: str = "whitegrid"):
        plt.style.use('default')
        sns.set_style(style)
        
    def plot_learning_curve(self, estimator, X, y, title: str = "Learning Curve",
                           cv: int = 5, train_sizes: Optional[np.ndarray] = None,
                           save_path: Optional[str] = None) -> None:
        """繪製學習曲線"""
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        # 計算學習曲線
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, train_sizes=train_sizes,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        # 轉換為正值（MSE）
        train_scores = -train_scores
        test_scores = -test_scores
        
        # 計算均值與標準差
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        # 繪圖
        plt.figure(figsize=(10, 6))
        plt.title(title, fontsize=16)
        plt.xlabel("Training examples", fontsize=14)
        plt.ylabel("MSE", fontsize=14)
        
        # 繪製訓練分數
        plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='blue',
                label="Training error", linewidth=2, markersize=6)
        plt.fill_between(train_sizes_abs, 
                        train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, 
                        alpha=0.1, color="blue")
        
        # 繪製驗證分數
        plt.plot(train_sizes_abs, test_scores_mean, 'o-', color='orange',
                label="Cross-validation error", linewidth=2, markersize=6)
        plt.fill_between(train_sizes_abs,
                        test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std,
                        alpha=0.1, color="orange")
        
        plt.legend(loc="best")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"學習曲線已保存至: {save_path}")
        
        plt.show()
    
    def analyze_overfitting(self, train_scores: np.ndarray, 
                          test_scores: np.ndarray) -> Dict[str, Any]:
        """分析過擬合情況"""
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        
        # 計算訓練與驗證分數的差距
        final_gap = abs(train_mean[-1] - test_mean[-1])
        avg_gap = np.mean(abs(train_mean - test_mean))
        
        # 判斷過擬合程度
        if final_gap < 0.1 * test_mean[-1]:
            overfitting_level = "低"
        elif final_gap < 0.2 * test_mean[-1]:
            overfitting_level = "中等"
        else:
            overfitting_level = "高"
        
        return {
            'final_gap': final_gap,
            'average_gap': avg_gap,
            'overfitting_level': overfitting_level,
            'final_train_score': train_mean[-1],
            'final_test_score': test_mean[-1]
        }
