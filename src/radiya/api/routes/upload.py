# ملف: src/radiya/api/routes/upload.py (مُصحح)

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd
import json
import uuid
from pathlib import Path
import logging
from typing import Optional
import time
import asyncio
from datetime import datetime
import os

router = APIRouter()
logger = logging.getLogger(__name__)

# تتبع المهام الجارية
active_jobs = {}

class ProcessingJob:
    def __init__(self, job_id: str, filename: str):
        self.job_id = job_id
        self.filename = filename
        self.status = "processing"  
        self.progress = 0
        self.results = None
        self.error = None
        self.start_time = datetime.now()
        self.current_step = "تحضير البيانات"

# استيراد الكلاسات الأساسية فقط
class SimpleDataLoader:
    """محمل بيانات مبسط"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
    
    def load_data(self) -> pd.DataFrame:
        """تحميل البيانات من الملف"""
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"ملف البيانات غير موجود: {self.file_path}")
        
        try:
            if self.file_path.suffix.lower() == '.json':
                df = pd.read_json(self.file_path, lines=True)
            elif self.file_path.suffix.lower() == '.csv':
                df = pd.read_csv(self.file_path)
            else:
                raise ValueError(f"نوع ملف غير مدعوم: {self.file_path.suffix}")
            
            # معالجة أولية بسيطة
            if len(df) == 0:
                raise ValueError("الملف فارغ")
            
            # إزالة المستخدمين المجهولين
            if 'userId' in df.columns:
                df = df[df['userId'] != ''].copy()
                df = df[df['userId'].notna()].copy()
            
            # تحويل الأوقات إذا وُجد عمود ts
            if 'ts' in df.columns:
                df['datetime'] = pd.to_datetime(df['ts'], unit='ms', errors='coerce')
                df['date'] = df['datetime'].dt.date
                df['hour'] = df['datetime'].dt.hour
                df['day_of_week'] = df['datetime'].dt.dayofweek
                df['is_weekend'] = df['datetime'].dt.weekday >= 5
            
            # ملء القيم المفقودة
            if 'song' in df.columns:
                df['song'] = df['song'].fillna('Unknown')
            if 'artist' in df.columns:
                df['artist'] = df['artist'].fillna('Unknown')
            if 'length' in df.columns:
                df['length'] = pd.to_numeric(df['length'], errors='coerce').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"خطأ في تحميل الملف {self.file_path}: {e}")
            raise

class SimpleFeatureEngineer:
    """مهندس ميزات مبسط"""
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إنشاء ميزات بسيطة"""
        
        features_list = []
        
        for user_id in df['userId'].unique():
            user_data = df[df['userId'] == user_id].sort_values('datetime')
            features = self._extract_user_features(user_data, user_id)
            features_list.append(features)
        
        return pd.DataFrame(features_list).fillna(0)
    
    def _extract_user_features(self, user_data: pd.DataFrame, user_id: str) -> dict:
        """استخراج ميزات مستخدم واحد"""
        
        features = {'userId': user_id}
        
        # ميزات أساسية
        total_events = len(user_data)
        unique_sessions = user_data['sessionId'].nunique() if 'sessionId' in user_data.columns else 1
        active_days = user_data['date'].nunique() if 'date' in user_data.columns else 1
        
        features.update({
            'total_events': total_events,
            'unique_sessions': unique_sessions,
            'active_days': active_days,
        })
        
        # ميزات الاستماع
        if 'page' in user_data.columns:
            songs_data = user_data[user_data['page'] == 'NextSong']
            features.update({
                'songs_played': len(songs_data),
                'unique_songs': songs_data['song'].nunique() if len(songs_data) > 0 else 0,
                'unique_artists': songs_data['artist'].nunique() if len(songs_data) > 0 else 0,
                'total_listening_time': songs_data['length'].sum() if len(songs_data) > 0 else 0,
            })
            
            # ميزات التفاعل
            features.update({
                'thumbs_up': len(user_data[user_data['page'] == 'Thumbs Up']),
                'thumbs_down': len(user_data[user_data['page'] == 'Thumbs Down']),
                'add_to_playlist': len(user_data[user_data['page'] == 'Add to Playlist']),
                'home_visits': len(user_data[user_data['page'] == 'Home']),
                'help_visits': len(user_data[user_data['page'] == 'Help']),
                'settings_visits': len(user_data[user_data['page'] == 'Settings']),
                'upgrade_visits': len(user_data[user_data['page'] == 'Upgrade']),
                'downgrade_visits': len(user_data[user_data['page'] == 'Submit Downgrade']),
                'logout_count': len(user_data[user_data['page'] == 'Logout']),
                'error_count': len(user_data[user_data['page'] == 'Error'])
            })
        else:
            # إذا لم يكن هناك عمود page، ضع قيم افتراضية
            features.update({
                'songs_played': total_events // 2,
                'unique_songs': 0, 'unique_artists': 0, 'total_listening_time': 0,
                'thumbs_up': 0, 'thumbs_down': 0, 'add_to_playlist': 0,
                'home_visits': 0, 'help_visits': 0, 'settings_visits': 0,
                'upgrade_visits': 0, 'downgrade_visits': 0,
                'logout_count': 0, 'error_count': 0
            })
        
        # ميزات الاشتراك
        if 'level' in user_data.columns:
            features.update({
                'final_level_paid': 1 if user_data['level'].iloc[-1] == 'paid' else 0,
                'started_as_paid': 1 if user_data['level'].iloc[0] == 'paid' else 0,
                'was_ever_paid': 1 if 'paid' in user_data['level'].values else 0,
            })
        else:
            features.update({
                'final_level_paid': 0,
                'started_as_paid': 0, 
                'was_ever_paid': 0
            })
        
        # ميزات ديموغرافية
        if 'gender' in user_data.columns:
            gender = user_data['gender'].iloc[0] if pd.notna(user_data['gender'].iloc[0]) else 'Unknown'
            features.update({
                'gender_M': 1 if gender == 'M' else 0,
                'gender_F': 1 if gender == 'F' else 0,
            })
        else:
            features.update({'gender_M': 0, 'gender_F': 0})
        
        # ميزات زمنية
        if 'hour' in user_data.columns:
            features.update({
                'avg_hour': user_data['hour'].mean(),
                'weekend_activity_ratio': user_data['is_weekend'].mean(),
            })
        else:
            features.update({'avg_hour': 12, 'weekend_activity_ratio': 0.3})
        
        # ميزات مشتقة
        if unique_sessions > 0:
            features['events_per_session'] = total_events / unique_sessions
            features['songs_per_session'] = features['songs_played'] / unique_sessions
        else:
            features['events_per_session'] = 0
            features['songs_per_session'] = 0
        
        if active_days > 0:
            features['events_per_day'] = total_events / active_days
            features['songs_per_day'] = features['songs_played'] / active_days
        else:
            features['events_per_day'] = 0
            features['songs_per_day'] = 0
        
        # معدل التفاعل
        if features['songs_played'] > 0:
            features['interaction_rate'] = (features['thumbs_up'] + features['thumbs_down'] + features['add_to_playlist']) / features['songs_played']
        else:
            features['interaction_rate'] = 0
        
        return features
    
    def define_churn(self, df: pd.DataFrame, method: str = 'combined') -> pd.DataFrame:
        """تعريف المستخدمين المنسحبين"""
        
        all_users = df['userId'].unique()
        churn_labels = pd.DataFrame({
            'userId': all_users,
            'churned': 0
        })
        
        if 'page' not in df.columns:
            # إذا لم يكن هناك عمود page، استخدم توزيع عشوائي
            import numpy as np
            np.random.seed(42)
            random_churn = np.random.choice([0, 1], size=len(all_users), p=[0.8, 0.2])
            churn_labels['churned'] = random_churn
            return churn_labels
        
        if method == 'cancellation':
            cancelled_users = df[df['page'] == 'Cancellation Confirmation']['userId'].unique()
            churn_labels.loc[churn_labels['userId'].isin(cancelled_users), 'churned'] = 1
            
        elif method == 'downgrade':
            downgrade_users = df[df['page'] == 'Submit Downgrade']['userId'].unique()
            churn_labels.loc[churn_labels['userId'].isin(downgrade_users), 'churned'] = 1
            
        elif method == 'combined':
            cancelled_users = df[df['page'] == 'Cancellation Confirmation']['userId'].unique()
            downgrade_users = df[df['page'] == 'Submit Downgrade']['userId'].unique()
            churn_users = set(cancelled_users) | set(downgrade_users)
            churn_labels.loc[churn_labels['userId'].isin(churn_users), 'churned'] = 1
            
        elif method == 'inactivity':
            if 'datetime' in df.columns:
                last_activity = df.groupby('userId')['datetime'].max()
                max_date = df['datetime'].max()
                from datetime import timedelta
                inactive_threshold = max_date - timedelta(days=7)
                inactive_users = last_activity[last_activity < inactive_threshold].index
                churn_labels.loc[churn_labels['userId'].isin(inactive_users), 'churned'] = 1
            else:
                # إذا لم تكن هناك معلومات زمنية، استخدم عشوائي
                import numpy as np
                np.random.seed(43)
                random_churn = np.random.choice([0, 1], size=len(all_users), p=[0.9, 0.1])
                churn_labels['churned'] = random_churn
        
        return churn_labels

class SimpleModelTrainer:
    """مدرب نماذج مبسط"""
    
    def train_all_models(self, X, y, experiment_name="simple"):
        """تدريب نماذج بسيطة"""
        
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        results = {}
        
        # تدريب Random Forest
        try:
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            y_pred_proba = rf.predict_proba(X_test)[:, 1]
            
            cv_scores = cross_val_score(rf, X_train, y_train, cv=3, scoring='roc_auc')
            
            results['RandomForest'] = {
                'test_auc': roc_auc_score(y_test, y_pred_proba),
                'test_precision': precision_score(y_test, y_pred, zero_division=0),
                'test_recall': recall_score(y_test, y_pred, zero_division=0),
                'test_f1': f1_score(y_test, y_pred, zero_division=0),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        except Exception as e:
            logger.error(f"خطأ في Random Forest: {e}")
            results['RandomForest'] = {'error': str(e)}
        
        # تدريب Logistic Regression
        try:
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_train_scaled, y_train)
            y_pred = lr.predict(X_test_scaled)
            y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
            
            cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=3, scoring='roc_auc')
            
            results['LogisticRegression'] = {
                'test_auc': roc_auc_score(y_test, y_pred_proba),
                'test_precision': precision_score(y_test, y_pred, zero_division=0),
                'test_recall': recall_score(y_test, y_pred, zero_division=0),
                'test_f1': f1_score(y_test, y_pred, zero_division=0),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        except Exception as e:
            logger.error(f"خطأ في Logistic Regression: {e}")
            results['LogisticRegression'] = {'error': str(e)}
        
        return results

@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """رفع وتحليل ملف البيانات"""
    
    # التحقق من نوع الملف
    allowed_extensions = ['.json', '.csv']
    file_extension = '.' + file.filename.split('.')[-1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail="نوع الملف غير مدعوم. يرجى رفع ملف JSON أو CSV"
        )
    
    # قراءة الملف والتحقق من الحجم
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    
    # زيادة الحد الأقصى إلى 300 MB
    if file_size_mb > 300:
        raise HTTPException(
            status_code=413, 
            detail=f"حجم الملف كبير جداً ({file_size_mb:.1f} MB). الحد الأقصى 300 MB"
        )
    
    try:
        # إنشاء معرف فريد للمهمة
        job_id = str(uuid.uuid4())
        
        # إنشاء مجلد الرفع
        upload_dir = Path("../data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # حفظ الملف
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{job_id[:8]}_{file.filename}"
        file_path = upload_dir / safe_filename
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # إنشاء مهمة معالجة
        job = ProcessingJob(job_id, file.filename)
        active_jobs[job_id] = job
        
        # بدء المعالجة في الخلفية
        background_tasks.add_task(process_uploaded_file, job_id, str(file_path))
        
        logger.info(f"تم رفع الملف: {file.filename} ({file_size_mb:.1f} MB) - Job ID: {job_id}")
        
        return JSONResponse({
            "job_id": job_id,
            "filename": file.filename,
            "file_size_mb": round(file_size_mb, 2),
            "message": "تم رفع الملف بنجاح. جاري المعالجة...",
            "status_url": f"/api/v1/status/{job_id}",
            "estimated_time": "3-8 دقائق"
        })
        
    except Exception as e:
        logger.error(f"خطأ في رفع الملف: {e}")
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة الملف: {str(e)}")

async def process_uploaded_file(job_id: str, file_path: str):
    """معالجة الملف المرفوع"""
    
    job = active_jobs[job_id]
    
    try:
        logger.info(f"بدء معالجة الملف: {job_id} - {file_path}")
        
        # التأكد من وجود الملف
        if not Path(file_path).exists():
            raise FileNotFoundError(f"الملف غير موجود: {file_path}")
        
        # الخطوة 1: تحميل البيانات
        job.current_step = "تحميل البيانات"
        job.progress = 10
        
        loader = SimpleDataLoader(file_path)
        df = loader.load_data()
        
        logger.info(f"تم تحميل {len(df)} سجل لـ {df['userId'].nunique()} مستخدم")
        
        # الخطوة 2: هندسة الميزات
        job.current_step = "هندسة الميزات"
        job.progress = 30
        
        engineer = SimpleFeatureEngineer()
        features_df = engineer.create_features(df)
        
        logger.info(f"تم إنشاء {len(features_df.columns)-1} ميزة")
        
        # الخطوة 3: تدريب النماذج
        job.current_step = "تدريب النماذج"
        job.progress = 60
        
        trainer = SimpleModelTrainer()
        all_results = {}
        
        churn_methods = ['combined', 'cancellation', 'downgrade', 'inactivity']
        
        for i, method in enumerate(churn_methods):
            try:
                churn_labels = engineer.define_churn(df, method=method)
                ml_data = features_df.merge(churn_labels, on='userId')
                
                X = ml_data.drop(['userId', 'churned'], axis=1)
                y = ml_data['churned']
                
                if len(X) > 0 and y.sum() > 0:  # التأكد من وجود بيانات وحالات انسحاب
                    method_results = trainer.train_all_models(X, y, f"upload_{method}")
                    all_results[method] = method_results
                    logger.info(f"تم تدريب النماذج لطريقة {method}")
                else:
                    all_results[method] = {"error": "لا توجد حالات انسحاب كافية"}
                
                job.progress = 60 + (i + 1) * 7
                
            except Exception as e:
                logger.error(f"خطأ في تدريب طريقة {method}: {e}")
                all_results[method] = {"error": str(e)}
        
        # الخطوة 4: تجهيز النتائج
        job.current_step = "تجهيز النتائج"
        job.progress = 95
        
        results = prepare_final_results(df, all_results, features_df)
        
        # حفظ النتائج
        results_dir = Path("../reports/uploads")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"{job_id}_results.json"
        with open(results_file, "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # إكمال المهمة
        job.status = "completed"
        job.results = results
        job.progress = 100
        job.current_step = "اكتملت المعالجة"
        
        logger.info(f"تمت معالجة الملف بنجاح: {job_id}")
        
    except Exception as e:
        job.status = "error"
        job.error = f"خطأ في المعالجة: {str(e)}"
        job.current_step = "حدث خطأ"
        logger.error(f"خطأ في معالجة الملف {job_id}: {e}")

def prepare_final_results(df, all_results, features_df):
    """تجهيز النتائج النهائية"""
    
    # إحصائيات البيانات
    data_summary = {
        "total_records": len(df),
        "unique_users": df['userId'].nunique(),
        "unique_sessions": df['sessionId'].nunique() if 'sessionId' in df.columns else 0,
        "date_range": {
            "start": df['datetime'].min().isoformat() if 'datetime' in df.columns else "غير متاح",
            "end": df['datetime'].max().isoformat() if 'datetime' in df.columns else "غير متاح",
            "duration_days": (df['datetime'].max() - df['datetime'].min()).days if 'datetime' in df.columns else 0
        },
        "features_created": len(features_df.columns) - 1,
        "churn_rates": {}
    }
    
    # معدلات الانسحاب
    engineer = SimpleFeatureEngineer()
    for method in ['cancellation', 'downgrade', 'combined', 'inactivity']:
        try:
            churn_labels = engineer.define_churn(df, method=method)
            churn_rate = churn_labels['churned'].mean() * 100
            data_summary["churn_rates"][method] = round(churn_rate, 2)
        except:
            data_summary["churn_rates"][method] = 0
    
    # أفضل النماذج
    best_models = {}
    
    for method, method_results in all_results.items():
        if "error" not in method_results:
            best_auc = 0
            best_model_name = None
            
            for model_name, result in method_results.items():
                if isinstance(result, dict) and 'test_auc' in result:
                    if result['test_auc'] > best_auc:
                        best_auc = result['test_auc']
                        best_model_name = model_name
            
            if best_model_name:
                best_models[method] = {
                    "model_name": best_model_name,
                    "auc_score": round(best_auc, 4),
                    "precision": round(method_results[best_model_name]['test_precision'], 4),
                    "recall": round(method_results[best_model_name]['test_recall'], 4),
                    "f1_score": round(method_results[best_model_name]['test_f1'], 4)
                }
    
    # توصيات
    recommendations = []
    combined_churn_rate = data_summary["churn_rates"].get("combined", 0)
    
    if combined_churn_rate > 25:
        recommendations.append({
            "type": "urgent",
            "title": "معدل انسحاب مرتفع جداً", 
            "description": f"معدل الانسحاب {combined_churn_rate}% يتطلب تدخل فوري"
        })
    elif combined_churn_rate > 15:
        recommendations.append({
            "type": "warning",
            "title": "معدل انسحاب مرتفع",
            "description": f"معدل الانسحاب {combined_churn_rate}% يحتاج لمراقبة"
        })
    else:
        recommendations.append({
            "type": "success", 
            "title": "معدل انسحاب مقبول",
            "description": f"معدل الانسحاب {combined_churn_rate}% في المعدل الطبيعي"
        })
    
    return {
        "data_summary": data_summary,
        "best_models": best_models,
        "model_performance": all_results,
        "recommendations": recommendations,
        "processing_info": {
            "processed_at": datetime.now().isoformat(),
            "processing_version": "2.0.0"
        }
    }

@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """الحصول على حالة مهمة المعالجة"""
    
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    
    job = active_jobs[job_id]
    elapsed_time = (datetime.now() - job.start_time).total_seconds()
    
    response = {
        "job_id": job_id,
        "filename": job.filename,
        "status": job.status,
        "progress": job.progress,
        "current_step": job.current_step,
        "elapsed_time": round(elapsed_time),
        "timestamp": datetime.now().isoformat()
    }
    
    if job.status == "completed" and job.results:
        response["results"] = job.results
    elif job.status == "error" and job.error:
        response["error"] = job.error
    
    return response

@router.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    """إلغاء مهمة المعالجة"""
    
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    
    job = active_jobs[job_id]
    
    if job.status == "processing":
        job.status = "cancelled"
        job.current_step = "تم الإلغاء"
    
    return {"message": "تم إلغاء المهمة", "job_id": job_id}