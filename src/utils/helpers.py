"""輔助工具函數"""

import os
import logging
from pathlib import Path
from typing import Optional

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """設定日誌記錄"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=numeric_level,
            format=format_string,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=numeric_level, format=format_string)

def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """驗證檔案路徑"""
    path = Path(file_path)
    
    if must_exist:
        return path.exists() and path.is_file()
    else:
        # 檢查父目錄是否存在
        return path.parent.exists()

def create_directory(dir_path: str) -> None:
    """建立目錄（如果不存在）"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def get_project_root() -> Path:
    """取得專案根目錄"""
    return Path(__file__).parent.parent.parent

def format_duration(minutes: float) -> str:
    """格式化手術時間顯示"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    
    if hours > 0:
        return f"{hours}小時{mins}分鐘"
    else:
        return f"{mins}分鐘"
