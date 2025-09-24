#!/usr/bin/env python3
"""模型評估腳本"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging

# 添加 src 到 path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.predictor import SurgeryPredictor
from src.data.loader import DataLoader
from src.evaluation.metrics import ModelEvaluator
from src.config.departments import DEPARTMENT_CONFIGS
from src.utils.helpers import setup_logging

def evaluate_department(department: str, data_loader: DataLoader, 
                       models_dir: str) -> dict:
    """評估單一科別模型"""
    logger = logging.getLogger(__name__)
    
    try:
        # 載入測試資料
        test_data = data_loader.load_testing_data(department)
        if test_data is None:
            return {}
        
        # 建立預測器
        predictor = SurgeryPredictor.from_department(department, models_dir)
        
        # 進行預測
        predictions = predictor.predict_batch(test_data)
        
        # 取得實際值
        y_true = test_data['手術時間（分）(BQ)'].values
        
        # 計算評估指標
        metrics = ModelEvaluator.calculate_metrics(y_true, predictions)
        
        logger.info(f"{department} 評估完成")
        return metrics
        
    except Exception as e:
        logger.error(f"評估 {department} 時發生錯誤: {str(e)}")
        return {}

def main():
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    # 設定路徑
    data_dir = "data"
    models_dir = "models"
    
    # 建立資料載入器
    data_loader = DataLoader(data_dir)
    
    # 評估結果
    all_results = {}
    
    logger.info("開始評估所有科別模型...")
    
    for dept_name in DEPARTMENT_CONFIGS.keys():
        logger.info(f"正在評估 {dept_name} 模型...")
        
        metrics = evaluate_department(dept_name, data_loader, models_dir)
        if metrics:
            all_results[dept_name] = metrics
            
            # 生成詳細報告
            report = ModelEvaluator.generate_evaluation_report(dept_name, metrics)
            print(report)
    
    if all_results:
        # 比較所有模型
        comparison_df = ModelEvaluator.compare_models(all_results)
        print("\n=== 模型比較結果 ===")
        print(comparison_df)
        
        # 保存結果
        comparison_df.to_csv("model_evaluation_results.csv")
        logger.info("評估結果已保存至: model_evaluation_results.csv")
    
    logger.info("所有模型評估完成")

if __name__ == "__main__":
    main()
