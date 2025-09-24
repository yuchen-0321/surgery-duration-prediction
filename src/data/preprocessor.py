"""資料前處理模組"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

from ..config.settings import FEATURE_CONFIG, DATA_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """資料前處理器"""
    
    def __init__(self):
        self.preprocessor: Optional[ColumnTransformer] = None
        self._setup_preprocessor()
    
    def _setup_preprocessor(self):
        """設定前處理器"""
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), FEATURE_CONFIG['numeric_features']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), 
                 FEATURE_CONFIG['categorical_features'])
            ]
        )
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清理資料"""
        cleaned_data = data.copy()
        
        # 處理目標變數中的 "?" 值
        target_col = DATA_CONFIG['target_column']
        if target_col in cleaned_data.columns:
            cleaned_data[target_col] = pd.to_numeric(
                cleaned_data[target_col].replace('?', np.nan), 
                errors='coerce'
            )
        
        # 填補數值型特徵缺失值
        for feature in FEATURE_CONFIG['numeric_features']:
            if feature in cleaned_data.columns:
                cleaned_data[feature] = cleaned_data[feature].fillna(
                    cleaned_data[feature].mean()
                )
        
        # 移除含有 NaN 的目標變數行
        if target_col in cleaned_data.columns:
            cleaned_data = cleaned_data.dropna(subset=[target_col])
        
        logger.info(f"資料清理完成。最終資料筆數: {len(cleaned_data)}")
        return cleaned_data
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """準備特徵與目標變數"""
        # 分離特徵與目標
        feature_columns = (FEATURE_CONFIG['numeric_features'] + 
                         FEATURE_CONFIG['categorical_features'])
        available_features = [col for col in feature_columns if col in data.columns]
        
        X = data[available_features]
        y = data[DATA_CONFIG['target_column']]
        
        return X, y
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """訓練並轉換特徵"""
        return self.preprocessor.fit_transform(X)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """轉換特徵（已訓練）"""
        if self.preprocessor is None:
            raise ValueError("前處理器尚未訓練，請先執行 fit_transform")
        return self.preprocessor.transform(X)
