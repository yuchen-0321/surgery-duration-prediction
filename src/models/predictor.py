"""手術時間預測器"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import joblib
import logging

from ..data.preprocessor import DataPreprocessor
from ..config.departments import DEPARTMENT_CONFIGS

logger = logging.getLogger(__name__)

class SurgeryPredictor:
    """手術時間預測器"""
    
    def __init__(self, department: str, model_path: Optional[str] = None):
        self.department = department
        self.model = None
        self.preprocessor = DataPreprocessor()
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """載入訓練好的模型"""
        try:
            self.model = joblib.load(model_path)
            logger.info(f"成功載入 {self.department} 模型")
        except Exception as e:
            logger.error(f"載入模型失敗: {str(e)}")
            raise
    
    def predict_single(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """預測單筆手術時間"""
        if self.model is None:
            raise ValueError("模型尚未載入")
        
        # 轉換為 DataFrame
        df = pd.DataFrame([patient_data])
        
        # 清理資料
        df = self.preprocessor.clean_data(df)
        
        # 預測
        prediction = self.model.predict(df)[0]
        
        return {
            "department": self.department,
            "predicted_duration_minutes": round(prediction, 1),
            "predicted_duration_hours": round(prediction / 60, 2),
            "patient_data": patient_data
        }
    
    def predict_batch(self, data: pd.DataFrame) -> np.ndarray:
        """批次預測"""
        if self.model is None:
            raise ValueError("模型尚未載入")
        
        # 清理資料
        cleaned_data = self.preprocessor.clean_data(data)
        
        # 預測
        predictions = self.model.predict(cleaned_data)
        
        logger.info(f"完成 {len(predictions)} 筆預測")
        return predictions
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """取得特徵重要性"""
        if self.model is None:
            return None
        
        try:
            # 取得隨機森林的特徵重要性
            regressor = self.model.named_steps['regressor']
            if hasattr(regressor, 'feature_importances_'):
                # 取得特徵名稱（需要從 preprocessor 取得）
                feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
                importance_dict = dict(zip(feature_names, regressor.feature_importances_))
                
                # 排序並返回前 10 個重要特徵
                sorted_importance = sorted(importance_dict.items(), 
                                         key=lambda x: x[1], reverse=True)[:10]
                return dict(sorted_importance)
        except Exception as e:
            logger.warning(f"無法取得特徵重要性: {str(e)}")
        
        return None
    
    @classmethod
    def from_department(cls, department: str, models_dir: str = "models") -> 'SurgeryPredictor':
        """從科別名稱建立預測器"""
        model_path = f"{models_dir}/{department}_random_forest_model.pkl"
        return cls(department, model_path)
