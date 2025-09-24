"""基礎訓練器抽象類別"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import logging

logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    """模型訓練器基礎類別"""
    
    def __init__(self, department: str):
        self.department = department
        self.model: Optional[Any] = None
        self.is_trained = False
        
    @abstractmethod
    def train(self, data_path: str) -> Dict[str, float]:
        """訓練模型"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """預測"""
        pass
    
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """評估模型"""
        pass
    
    def save_model(self, save_path: str) -> None:
        """保存模型"""
        if not self.is_trained:
            raise ValueError("模型尚未訓練，無法保存")
        
        joblib.dump(self.model, save_path)
        logger.info(f"{self.department} 模型已保存至: {save_path}")
    
    def load_model(self, model_path: str) -> None:
        """載入模型"""
        try:
            self.model = joblib.load(model_path)
            self.is_trained = True
            logger.info(f"{self.department} 模型已從 {model_path} 載入")
        except Exception as e:
            logger.error(f"載入 {self.department} 模型時發生錯誤: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """取得模型資訊"""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "department": self.department,
            "model_type": type(self.model).__name__,
            "is_trained": self.is_trained,
            "status": "ready"
        }
