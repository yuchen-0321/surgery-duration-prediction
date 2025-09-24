#!/usr/bin/env python3
"""超參數調整腳本"""

import sys
import os
from pathlib import Path
import argparse
import logging

# 添加 src 到 path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.random_forest_trainer import RandomForestTrainer
from src.config.departments import DEPARTMENT_CONFIGS
from src.utils.helpers import setup_logging

def main():
    parser = argparse.ArgumentParser(description="手術時間預測模型超參數調整")
    parser.add_argument("--department", type=str, required=True,
                       choices=list(DEPARTMENT_CONFIGS.keys()),
                       help="指定科別")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                       help="資料目錄路徑")
    parser.add_argument("--model-dir", type=str, default="models",
                       help="模型保存目錄")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日誌等級")
    
    args = parser.parse_args()
    
    # 設定日誌
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"開始調整 {args.department} 模型超參數...")
    
    # 建立訓練器（啟用超參數調整）
    trainer = RandomForestTrainer(args.department, use_hyperparameter_tuning=True)
    
    # 訓練資料路徑
    data_file = os.path.join(args.data_dir, f"{args.department}_Training_Cleaned.csv")
    
    if not os.path.exists(data_file):
        logger.error(f"找不到訓練資料: {data_file}")
        return
    
    try:
        # 執行超參數調整與訓練
        metrics = trainer.train(data_file)
        
        # 保存最佳模型
        os.makedirs(args.model_dir, exist_ok=True)
        model_file = os.path.join(args.model_dir, 
                                 f"{args.department}_tuned_model.pkl")
        trainer.save_model(model_file)
        
        # 輸出結果
        logger.info("超參數調整完成！")
        logger.info(f"最佳模型效能 - MSE: {metrics['mse']:.2f}, "
                   f"MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")
        
    except Exception as e:
        logger.error(f"超參數調整過程發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main()
