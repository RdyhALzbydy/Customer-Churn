import pandas as pd
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class DataValidator:
    """فئة التحقق من صحة البيانات"""
    
    REQUIRED_COLUMNS = ['userId', 'sessionId', 'page', 'ts', 'level']
    SPARKIFY_PAGES = [
        'NextSong', 'Home', 'Upgrade', 'Thumbs Up', 'Thumbs Down',
        'Add to Playlist', 'Add Friend', 'Settings', 'Help', 'Logout',
        'Submit Downgrade', 'Cancellation Confirmation', 'Error'
    ]
    
    def validate_sparkify_data(self, df: pd.DataFrame) -> ValidationResult:
        """التحقق من صحة بيانات Sparkify"""
        
        errors = []
        warnings = []
        
        # التحقق من الأعمدة المطلوبة
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            errors.append(f"أعمدة مفقودة: {', '.join(missing_cols)}")
        
        # التحقق من وجود بيانات
        if len(df) == 0:
            errors.append("الملف فارغ")
        
        # التحقق من المستخدمين
        if 'userId' in df.columns:
            empty_users = df['userId'].isna().sum() + (df['userId'] == '').sum()
            if empty_users > len(df) * 0.5:
                errors.append("أكثر من 50% من البيانات بدون معرف مستخدم")
            elif empty_users > 0:
                warnings.append(f"{empty_users} سجل بدون معرف مستخدم")
        
        # التحقق من صيغة التوقيت
        if 'ts' in df.columns:
            try:
                pd.to_datetime(df['ts'], unit='ms')
            except:
                errors.append("صيغة التوقيت (ts) غير صحيحة")
        
        # التحقق من صفحات Sparkify
        if 'page' in df.columns:
            unknown_pages = set(df['page'].dropna()) - set(self.SPARKIFY_PAGES)
            if unknown_pages:
                warnings.append(f"صفحات غير معروفة: {', '.join(list(unknown_pages)[:5])}")
        
        # التحقق من مستويات الاشتراك
        if 'level' in df.columns:
            valid_levels = {'free', 'paid'}
            invalid_levels = set(df['level'].dropna()) - valid_levels
            if invalid_levels:
                warnings.append(f"مستويات اشتراك غير معروفة: {', '.join(invalid_levels)}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings)