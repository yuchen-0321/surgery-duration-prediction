"""資料載入模組"""

import pandas as pd
import os
from typing import Optional, List
import logging

from ..config.settings import DATA_CONFIG

logger = logging.getLogger(__name__)

class DataLoader:
    """資料載入器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        
    def load_training_data(self, department: str) -> Optional[pd.DataFrame]:
        """載入訓練資料"""
        filename = f"{department}_Training_Cleaned.csv"
        filepath = os.path.join(self.data_dir, "processed", filename)
        
        if not os.path.exists(filepath):
            logger.error(f"找不到訓練資料檔案: {filepath}")
            return None
            
        try:
            data = pd.read_csv(filepath, encoding=DATA_CONFIG['encoding'])
            logger.info(f"成功載入 {department} 訓練資料，共 {len(data)} 筆記錄")
            return data
        except Exception as e:
            logger.error(f"載入 {department} 訓練資料時發生錯誤: {str(e)}")
            return None
    
    def load_testing_data(self, department: str) -> Optional[pd.DataFrame]:
        """載入測試資料"""
        filename = f"{department}_Testing.csv"
        filepath = os.path.join(self.data_dir, "raw", filename)
        
        if not os.path.exists(filepath):
            logger.error(f"找不到測試資料檔案: {filepath}")
            return None
            
        try:
            data = pd.read_csv(filepath, encoding=DATA_CONFIG['encoding'])
            logger.info(f"成功載入 {department} 測試資料，共 {len(data)} 筆記錄")
            return data
        except Exception as e:
            logger.error(f"載入 {department} 測試資料時發生錯誤: {str(e)}")
            return None
    
    def list_available_departments(self) -> List[str]:
        """列出可用的科別資料"""
        departments = []
        processed_dir = os.path.join(self.data_dir, "processed")
        
        if not os.path.exists(processed_dir):
            logger.warning(f"處理後資料目錄不存在: {processed_dir}")
            return departments
        
        for filename in os.listdir(processed_dir):
            if filename.endswith("_Training_Cleaned.csv"):
                dept = filename.replace("_Training_Cleaned.csv", "")
                departments.append(dept)
        
        logger.info(f"找到 {len(departments)} 個科別的訓練資料: {departments}")
        return departments
    
    def validate_data_structure(self, data: pd.DataFrame) -> bool:
        """驗證資料結構"""
        required_columns = (DATA_CONFIG.get('required_columns', []) + 
                          [DATA_CONFIG['target_column']])
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"資料缺少必要欄位: {missing_columns}")
            return False
        
        logger.info("資料結構驗證通過")
        return True
