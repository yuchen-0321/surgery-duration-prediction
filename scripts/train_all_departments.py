#!/usr/bin/env python3
"""訓練所有部門模型的腳本"""

import os
import sys
from pathlib import Path
import pandas as pd
import logging

# 添加 src 到 path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.random_forest_trainer import RandomForestTrainer
from src.config.departments import DEPARTMENT_CONFIGS
from src.config.settings import DATA_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_department_model(department: str, data_dir: str, model_dir: str, 
                         use_tuning: bool = False) -> dict:
    """訓練單一部門模型"""
    
    # 構建檔案路徑
    data_file = os.path.join(data_dir, f"{department}_Training_Cleaned.csv")
    model_file = os.path.join(model_dir, f"{department}_random_forest_model.pkl")
    
    if not os.path.exists(data_file):
        logger.error(f"找不到訓練資料: {data_file}")
        return {}
    
    try:
        # 建立訓練器
        trainer = RandomForestTrainer(department, use_hyperparameter_tuning=use_tuning)
        
        # 訓練模型
        metrics = trainer.train(data_file)
        
        # 保存模型
        os.makedirs(model_dir, exist_ok=True)
        trainer.save_model(model_file)
        
        return metrics
        
    except Exception as e:
        logger.error(f"訓練 {department} 模型時發生錯誤: {str(e)}")
        return {}

def main():
    """主函數"""
    # 設定路徑
    data_dir = "data/processed"  # 修改為你的資料路徑
    model_dir = "models"
    
    # 訓練結果統計
    results = {}
    
    logger.info("開始訓練所有部門模型...")
    
    # 訓練所有部門
    for dept_name, dept_config in DEPARTMENT_CONFIGS.items():
        logger.info(f"正在訓練 {dept_name} ({dept_config.display_name}) 模型...")
        
        metrics = train_department_model(
            department=dept_name,
            data_dir=data_dir,
            model_dir=model_dir,
            use_tuning=False  # 使用預設最佳參數
        )
        
        if metrics:
            results[dept_name] = metrics
            logger.info(f"{dept_name} 訓練完成 - "
                       f"MSE: {metrics['mse']:.2f}, "
                       f"MAE: {metrics['mae']:.2f}, "
                       f"R²: {metrics['r2']:.3f}")
    
    # 輸出總結
    logger.info("\n=== 訓練完成總結 ===")
    for dept, metrics in results.items():
        logger.info(f"{dept}: MSE={metrics['mse']:.2f}, "
                   f"MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")

if __name__ == "__main__":
    main()
