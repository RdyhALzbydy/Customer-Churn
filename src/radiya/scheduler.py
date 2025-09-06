"""
مجدول إعادة التدريب التلقائي لمشروع رضية
"""

import logging
import os
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
from typing import Dict, Any

import pandas as pd
from radiya.features.engineer import SimpleFeatureEngineer
from radiya.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)

class RetrainingScheduler:
    """مجدول إعادة التدريب التلقائي للنماذج"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        تهيئة المجدول
        
        Args:
            config: إعدادات المجدول
        """
        self.config = config or {
            'retraining_frequency': 'weekly',  # daily, weekly, monthly
            'min_new_data_threshold': 1000,    # الحد الأدنى للبيانات الجديدة
            'performance_threshold': 0.8,      # عتبة الأداء لإعادة التدريب
            'data_drift_threshold': 0.1,       # عتبة انحراف البيانات
            'backup_models': True,              # نسخ احتياطية للنماذج
            'notification_enabled': True        # تفعيل الإشعارات
        }
        self.engineer = SimpleFeatureEngineer()
        self.trainer = ModelTrainer()
        self.is_running = False
        self.last_training_time = None
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    def start_scheduler(self):
        """تشغيل المجدول"""
        if self.is_running:
            logger.warning("المجدول يعمل بالفعل")
            return
            
        self.is_running = True
        logger.info(f"تم تشغيل مجدول إعادة التدريب - التكرار: {self.config['retraining_frequency']}")
        
        # جدولة المهام
        if self.config['retraining_frequency'] == 'daily':
            schedule.every().day.at("02:00").do(self._check_and_retrain)
        elif self.config['retraining_frequency'] == 'weekly':
            schedule.every().sunday.at("02:00").do(self._check_and_retrain)
        elif self.config['retraining_frequency'] == 'monthly':
            schedule.every(30).days.at("02:00").do(self._check_and_retrain)
        
        # تشغيل في خيط منفصل
        scheduler_thread = Thread(target=self._run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
    def stop_scheduler(self):
        """إيقاف المجدول"""
        self.is_running = False
        schedule.clear()
        logger.info("تم إيقاف مجدول إعادة التدريب")
        
    def _run_scheduler(self):
        """تشغيل حلقة المجدول"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # فحص كل دقيقة
            
    def _check_and_retrain(self):
        """فحص الحاجة لإعادة التدريب وتنفيذها"""
        try:
            logger.info("بدء فحص الحاجة لإعادة التدريب")
            
            # فحص البيانات الجديدة
            if not self._has_sufficient_new_data():
                logger.info("لا توجد بيانات جديدة كافية لإعادة التدريب")
                return
                
            # فحص أداء النماذج الحالية
            if not self._needs_retraining():
                logger.info("النماذج الحالية تؤدي بشكل جيد، لا حاجة لإعادة التدريب")
                return
                
            # تنفيذ إعادة التدريب
            self._perform_retraining()
            
        except Exception as e:
            logger.error(f"خطأ في عملية فحص إعادة التدريب: {e}")
            if self.config['notification_enabled']:
                self._send_notification(f"فشل في إعادة التدريب: {e}", level="error")
                
    def _has_sufficient_new_data(self) -> bool:
        """فحص وجود بيانات جديدة كافية"""
        data_dir = Path("data/raw")
        if not data_dir.exists():
            return False
            
        # البحث عن ملفات البيانات الجديدة
        new_files = []
        cutoff_time = datetime.now() - timedelta(days=1)
        
        for file_path in data_dir.glob("*.json"):
            if file_path.stat().st_mtime > cutoff_time.timestamp():
                new_files.append(file_path)
                
        if not new_files:
            return False
            
        # حساب حجم البيانات الجديدة
        total_new_records = 0
        for file_path in new_files:
            try:
                df = pd.read_json(file_path)
                total_new_records += len(df)
            except Exception as e:
                logger.warning(f"خطأ في قراءة الملف {file_path}: {e}")
                continue
                
        return total_new_records >= self.config['min_new_data_threshold']
        
    def _needs_retraining(self) -> bool:
        """فحص الحاجة لإعادة التدريب بناءً على الأداء"""
        # فحص آخر تدريب
        if self.last_training_time:
            time_since_last = datetime.now() - self.last_training_time
            if time_since_last.days < 7:  # لا نعيد التدريب قبل أسبوع
                return False
                
        # فحص أداء النماذج الحالية (محاكاة)
        current_performance = self._evaluate_current_models()
        
        return current_performance < self.config['performance_threshold']
        
    def _evaluate_current_models(self) -> float:
        """تقييم أداء النماذج الحالية"""
        # محاكاة تقييم الأداء - في التطبيق الحقيقي نحتاج لبيانات اختبار حديثة
        try:
            # قراءة آخر نتائج التدريب
            results_dir = Path("reports/metrics")
            if not results_dir.exists():
                return 0.0
                
            latest_results = None
            latest_time = 0
            
            for file_path in results_dir.glob("experiment_log_*.json"):
                if file_path.stat().st_mtime > latest_time:
                    latest_time = file_path.stat().st_mtime
                    latest_results = file_path
                    
            if latest_results:
                import json
                with open(latest_results, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('best_overall_model', {}).get('score', 0.0)
                
        except Exception as e:
            logger.warning(f"خطأ في تقييم النماذج الحالية: {e}")
            
        return 0.5  # قيمة افتراضية
        
    def _perform_retraining(self):
        """تنفيذ عملية إعادة التدريب"""
        logger.info("بدء عملية إعادة التدريب")
        
        try:
            # نسخ احتياطية للنماذج الحالية
            if self.config['backup_models']:
                self._backup_current_models()
                
            # تحميل البيانات الجديدة
            df = self._load_latest_data()
            if df is None or len(df) == 0:
                logger.error("فشل في تحميل البيانات")
                return
                
            # هندسة الميزات
            features_df = self.engineer.engineer_features(df)
            
            # التدريب لكل طريقة تعريف الانسحاب
            methods = ['cancellation', 'downgrade', 'combined', 'inactivity']
            results = {}
            
            for method in methods:
                logger.info(f"بدء التدريب للطريقة: {method}")
                
                # تعريف الانسحاب
                churn_labels = self.engineer.define_churn(df, method=method)
                
                # دمج الميزات مع التسميات
                final_df = features_df.merge(churn_labels[['userId', 'churned']], 
                                           on='userId', how='inner')
                
                if len(final_df) == 0:
                    logger.warning(f"لا توجد بيانات للطريقة {method}")
                    continue
                    
                # فصل المتغيرات
                X = final_df.drop(['userId', 'churned'], axis=1)
                y = final_df['churned']
                
                # التدريب
                method_results = self.trainer.train_all_models(
                    X, y, 
                    experiment_name=f"automated_retraining_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                results[method] = method_results
                
                logger.info(f"اكتمل التدريب للطريقة {method}")
                
            # حفظ النتائج
            self._save_retraining_results(results)
            
            # تحديث وقت آخر تدريب
            self.last_training_time = datetime.now()
            
            # إرسال إشعار النجاح
            if self.config['notification_enabled']:
                self._send_notification("تمت عملية إعادة التدريب بنجاح", level="info")
                
            logger.info("اكتملت عملية إعادة التدريب بنجاح")
            
        except Exception as e:
            logger.error(f"خطأ في عملية إعادة التدريب: {e}")
            if self.config['backup_models']:
                self._restore_backup_models()
            raise
            
    def _backup_current_models(self):
        """نسخ احتياطي للنماذج الحالية"""
        backup_dir = self.models_dir / "backups" / datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for model_file in self.models_dir.glob("*.pkl"):
            import shutil
            shutil.copy2(model_file, backup_dir)
            
        logger.info(f"تم إنشاء نسخة احتياطية في: {backup_dir}")
        
    def _restore_backup_models(self):
        """استعادة النماذج من النسخة الاحتياطية"""
        backup_dirs = list((self.models_dir / "backups").glob("*"))
        if not backup_dirs:
            logger.error("لا توجد نسخ احتياطية متاحة")
            return
            
        # أحدث نسخة احتياطية
        latest_backup = max(backup_dirs, key=lambda x: x.stat().st_mtime)
        
        import shutil
        for backup_file in latest_backup.glob("*.pkl"):
            shutil.copy2(backup_file, self.models_dir)
            
        logger.info(f"تم استعادة النماذج من النسخة الاحتياطية: {latest_backup}")
        
    def _load_latest_data(self) -> pd.DataFrame:
        """تحميل أحدث البيانات"""
        data_dir = Path("data/raw")
        
        # البحث عن أحدث ملف بيانات
        data_files = list(data_dir.glob("*.json"))
        if not data_files:
            return None
            
        latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"تحميل البيانات من: {latest_file}")
        
        return pd.read_json(latest_file)
        
    def _save_retraining_results(self, results: Dict[str, Any]):
        """حفظ نتائج إعادة التدريب"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = Path(f"reports/metrics/automated_retraining_{timestamp}.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"تم حفظ نتائج إعادة التدريب في: {results_file}")
        
    def _send_notification(self, message: str, level: str = "info"):
        """إرسال إشعار"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] {level.upper()}: {message}"
        
        # كتابة في السجل
        if level == "error":
            logger.error(formatted_message)
        elif level == "warning":
            logger.warning(formatted_message)
        else:
            logger.info(formatted_message)
            
        # يمكن إضافة إرسال بريد إلكتروني أو Slack هنا
        
    def get_status(self) -> Dict[str, Any]:
        """الحصول على حالة المجدول"""
        return {
            'is_running': self.is_running,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'next_scheduled_run': self._get_next_run_time(),
            'config': self.config
        }
        
    def _get_next_run_time(self) -> str:
        """الحصول على وقت التشغيل التالي المجدول"""
        if not schedule.jobs:
            return "غير مجدول"
            
        next_run = schedule.next_run()
        if next_run:
            return next_run.isoformat()
        return "غير محدد"

# مثيل عام للمجدول
retraining_scheduler = RetrainingScheduler()

def start_automated_retraining(config: Dict[str, Any] = None):
    """تشغيل نظام إعادة التدريب التلقائي"""
    global retraining_scheduler
    
    if config:
        retraining_scheduler.config.update(config)
        
    retraining_scheduler.start_scheduler()
    
def stop_automated_retraining():
    """إيقاف نظام إعادة التدريب التلقائي"""
    global retraining_scheduler
    retraining_scheduler.stop_scheduler()
    
def get_retraining_status():
    """الحصول على حالة نظام إعادة التدريب"""
    global retraining_scheduler
    return retraining_scheduler.get_status()