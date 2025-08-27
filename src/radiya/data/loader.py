"""
محمل البيانات لمشروع رضية
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DataLoader:
    """كلاس تحميل ومعالجة بيانات Sparkify"""
    
    def __init__(self, data_path: str):
        """
        تهيئة محمل البيانات
        
        Args:
            data_path: مسار ملف البيانات
        """
        data_path = "data/raw/customer_churn_mini.json"
        self.data_path = Path(data_path)
        self.df = None
        self.metadata = {}
        
    def load_data(self) -> pd.DataFrame:
        """تحميل البيانات من الملف"""
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"ملف البيانات غير موجود: {self.data_path}")
        
        logger.info(f"تحميل البيانات من: {self.data_path}")
        
        try:
            # تحديد نوع الملف وطريقة القراءة
            if self.data_path.suffix == '.json':
                self.df = pd.read_json(self.data_path, lines=True)
            elif self.data_path.suffix == '.csv':
                self.df = pd.read_csv(self.data_path)
            else:
                raise ValueError(f"نوع ملف غير مدعوم: {self.data_path.suffix}")
            
            logger.info(f"تم تحميل {len(self.df):,} سجل")
            
            # حفظ معلومات أساسية
            self.metadata = {
                'total_records': len(self.df),
                'columns': list(self.df.columns),
                'file_size_mb': self.data_path.stat().st_size / 1024 / 1024
            }
            
            # معالجة أولية
            self._initial_processing()
            
            return self.df
            
        except Exception as e:
            logger.error(f"خطأ في تحميل البيانات: {e}")
            raise
    
    def _initial_processing(self):
        """معالجة أولية للبيانات"""
        
        logger.info("معالجة أولية للبيانات...")
        
        # تنظيف البيانات
        initial_count = len(self.df)
        
        # إزالة المستخدمين المجهولين
        self.df = self.df[self.df['userId'] != ''].copy()
        removed_count = initial_count - len(self.df)
        
        if removed_count > 0:
            logger.info(f"تم حذف {removed_count:,} سجل للمستخدمين المجهولين")
        
        # تحويل الأوقات
        self.df['datetime'] = pd.to_datetime(self.df['ts'], unit='ms')
        self.df['date'] = self.df['datetime'].dt.date
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
        self.df['is_weekend'] = self.df['datetime'].dt.weekday >= 5
        
        # تحويل الأنواع
        self.df['userId'] = self.df['userId'].astype(str)
        self.df['length'] = pd.to_numeric(self.df['length'], errors='coerce')
        
        # معالجة القيم المفقودة
        self.df['song'] = self.df['song'].fillna('Unknown')
        self.df['artist'] = self.df['artist'].fillna('Unknown')
        self.df['length'] = self.df['length'].fillna(0)
        
        # حفظ إحصائيات محدثة
        self.metadata.update({
            'clean_records': len(self.df),
            'unique_users': self.df['userId'].nunique(),
            'date_range': (self.df['datetime'].min(), self.df['datetime'].max()),
            'total_songs': len(self.df[self.df['page'] == 'NextSong']),
            'unique_pages': self.df['page'].nunique()
        })
        
        logger.info(f"البيانات المعالجة: {self.df['userId'].nunique():,} مستخدم، {len(self.df):,} سجل")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """إرجاع ملخص البيانات"""
        
        if self.df is None:
            return {}
        
        # إحصائيات أساسية
        summary = {
            'basic_stats': self.metadata,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict(),
            'unique_values': {col: self.df[col].nunique() for col in self.df.columns},
        }
        
        # إحصائيات خاصة بـ Sparkify
        if 'page' in self.df.columns:
            summary['page_distribution'] = self.df['page'].value_counts().to_dict()
        
        if 'level' in self.df.columns:
            summary['subscription_levels'] = self.df.groupby('userId')['level'].last().value_counts().to_dict()
        
        if 'gender' in self.df.columns:
            summary['gender_distribution'] = self.df.groupby('userId')['gender'].first().value_counts().to_dict()
        
        return summary
    
    def save_processed_data(self, output_path: str):
        """حفظ البيانات المعالجة"""
        
        if self.df is None:
            raise ValueError("لا توجد بيانات لحفظها. قم بتحميل البيانات أولاً.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        logger.info(f"تم حفظ البيانات المعالجة في: {output_path}")
    
    def get_user_sample(self, n_users: int = 1000) -> pd.DataFrame:
        """الحصول على عينة من المستخدمين للتطوير السريع"""
        
        if self.df is None:
            raise ValueError("لا توجد بيانات. قم بتحميل البيانات أولاً.")
        
        # اختيار عينة عشوائية من المستخدمين
        sample_users = np.random.choice(
            self.df['userId'].unique(), 
            size=min(n_users, self.df['userId'].nunique()),
            replace=False
        )
        
        sample_df = self.df[self.df['userId'].isin(sample_users)].copy()
        logger.info(f"تم إنشاء عينة من {len(sample_users)} مستخدم ({len(sample_df):,} سجل)")
        
        return sample_df