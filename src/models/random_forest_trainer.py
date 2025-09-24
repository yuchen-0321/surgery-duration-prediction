"""隨機森林訓練器"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging

from ..data.preprocessor import DataPreprocessor
from ..config.settings import MODEL_CONFIG, CV_CONFIG, DATA_CONFIG
from ..config.departments import DEPARTMENT_CONFIGS

logger = logging.getLogger(__name__)

class RandomForestTrainer:
    """隨機森林訓練器"""
    
    def __init__(self, department: str, use_hyperparameter_tuning: bool = False):
        self.department = department
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.preprocessor = DataPreprocessor()
        self.model: Optional[Pipeline] = None
        self.is_trained = False
        
    def _create_pipeline(self, params: Dict[str, Any]) -> Pipeline:
        """建立訓練 Pipeline"""
        rf_params = {**MODEL_CONFIG['random_forest']['default_params'], **params}
        
        return Pipeline([
            ('preprocessor', self.preprocessor.preprocessor),
            ('regressor', RandomForestRegressor(**rf_params))
        ])
    
    def _get_best_params(self) -> Dict[str, Any]:
        """取得最佳參數"""
        if self.department in DEPARTMENT_CONFIGS:
            return DEPARTMENT_CONFIGS[self.department].best_params
        else:
            logger.warning(f"未找到 {self.department} 的最佳參數，使用預設值")
            return MODEL_CONFIG['random_forest']['default_params']
    
    def train(self, data_path: str) -> Dict[str, float]:
        """訓練模型"""
        logger.info(f"開始訓練 {self.department} 模型...")
        
        # 載入與清理資料
        data = pd.read_csv(data_path, encoding=DATA_CONFIG['encoding'])
        data = self.preprocessor.clean_data(data)
        
        # 準備特徵
        X, y = self.preprocessor.prepare_features(data)
        
        if self.use_hyperparameter_tuning:
            self.model = self._hyperparameter_tuning(X, y)
        else:
            # 使用預設最佳參數
            best_params = self._get_best_params()
            self.model = self._create_pipeline(best_params)
            self.model.fit(X, y)
        
        self.is_trained = True
        
        # 評估性能
        metrics = self.evaluate(X, y)
        logger.info(f"{self.department} 訓練完成。R²: {metrics['r2']:.3f}")
        
        return metrics
    
    def _hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        """超參數調整"""
        logger.info(f"開始 {self.department} 超參數調整...")
        
        # 建立基礎 Pipeline
        base_pipeline = self._create_pipeline({})
        
        # 設定參數搜索空間（加上 regressor__ 前綴）
        param_grid = {
            f'regressor__{key}': value 
            for key, value in MODEL_CONFIG['random_forest']['param_grid'].items()
        }
        
        # 隨機搜索
        random_search = RandomizedSearchCV(
            base_pipeline,
            param_distributions=param_grid,
            n_iter=CV_CONFIG['n_iter'],
            cv=CV_CONFIG['cv_folds'],
            scoring=CV_CONFIG['scoring'],
            verbose=CV_CONFIG['verbose'],
            n_jobs=-1,
            random_state=DATA_CONFIG['random_state']
        )
        
        random_search.fit(X, y)
        
        logger.info(f"最佳參數: {random_search.best_params_}")
        logger.info(f"最佳 MSE: {-random_search.best_score_:.2f}")
        
        return random_search.best_estimator_
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """評估模型性能"""
        if not self.is_trained:
            raise ValueError("模型尚未訓練")
        
        y_pred = self.model.predict(X)
        
        return {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
    
    def save_model(self, save_path: str):
        """保存模型"""
        if not self.is_trained:
            raise ValueError("模型尚未訓練")
        
        joblib.dump(self.model, save_path)
        logger.info(f"模型已保存至: {save_path}")
    
    def load_model(self, model_path: str):
        """載入模型"""
        self.model = joblib.load(model_path)
        self.is_trained = True
        logger.info(f"模型已從 {model_path} 載入")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """預測"""
        if not self.is_trained:
            raise ValueError("模型尚未訓練或載入")
        
        return self.model.predict(X)
