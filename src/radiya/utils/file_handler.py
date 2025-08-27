import pandas as pd
import json
from pathlib import Path
from typing import Union, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FileHandler:
    """معالج الملفات المرفوعة"""
    
    @staticmethod
    def read_uploaded_file(file_path: Union[str, Path]) -> pd.DataFrame:
        """قراءة الملف المرفوع"""
        
        file_path = Path(file_path)
        
        try:
            if file_path.suffix.lower() == '.json':
                return pd.read_json(file_path, lines=True)
            elif file_path.suffix.lower() == '.csv':
                return pd.read_csv(file_path)
            else:
                raise ValueError(f"نوع ملف غير مدعوم: {file_path.suffix}")
                
        except Exception as e:
            logger.error(f"خطأ في قراءة الملف {file_path}: {e}")
            raise
    
    @staticmethod
    def validate_file_size(file_path: Union[str, Path], max_size_mb: int = 129) -> bool:
        """التحقق من حجم الملف"""
        
        file_path = Path(file_path)
        size_mb = file_path.stat().st_size / (1024 * 1024)
        return size_mb <= max_size_mb
    
    @staticmethod
    def clean_uploaded_files(older_than_hours: int = 24):
        """تنظيف الملفات المرفوعة القديمة"""
        
        upload_dir = Path("data/uploads")
        if not upload_dir.exists():
            return
        
        import time
        current_time = time.time()
        cutoff_time = current_time - (older_than_hours * 3600)
        
        for file_path in upload_dir.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                logger.info(f"تم حذف الملف القديم: {file_path}")