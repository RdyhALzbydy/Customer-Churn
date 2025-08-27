"""
مدرب النماذج لمشروع رضية
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)

# النماذج
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# معالجة عدم التوازن
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

import joblib
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelTrainer:
    """كلاس تدريب النماذج"""
    
    def __init__(self, random_state: int = 42):
        """
        تهيئة مدرب النماذج
        
        Args:
            random_state: بذرة العشوائية لضمان إعادة الإنتاج
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.best_model = None
        self.best_score = 0
        
    def get_models_config(self) -> Dict[str, Dict]:
        """إعدادات النماذج المختلفة"""
        
        return {
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'scale_features': False,
                'description': 'Random Forest - نموذج قوي ومقاوم للإفراط في التدريب'
            },
            
            'LogisticRegression': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    solver='lbfgs'
                ),
                'scale_features': True,
                'description': 'Logistic Regression - نموذج خطي بسيط وقابل للتفسير'
            },
            
            'GradientBoosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=self.random_state
                ),
                'scale_features': False,
                'description': 'Gradient Boosting - نموذج تدرجي قوي'
            },
            
            'SVM': {
                'model': SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=self.random_state
                ),
                'scale_features': True,
                'description': 'Support Vector Machine - قوي مع البيانات عالية الأبعاد'
            }
        }
    
    def handle_imbalanced_data(self, X: pd.DataFrame, y: pd.Series, 
                              method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        معالجة عدم توازن البيانات
        
        Args:
            X: الميزات
            y: المتغير التابع
            method: طريقة المعالجة
            
        Returns:
            البيانات المتوازنة
        """
        
        logger.info(f"معالجة عدم التوازن باستخدام: {method}")
        original_distribution = y.value_counts()
        logger.info(f"التوزيع الأصلي: {original_distribution.to_dict()}")
        
        try:
            if method == 'smote':
                # التأكد من وجود عينات كافية
                min_class_count = min(y.value_counts())
                k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
                sampler = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
                
            elif method == 'adasyn':
                sampler = ADASYN(random_state=self.random_state)
                
            elif method == 'smotetomek':
                sampler = SMOTETomek(random_state=self.random_state)
                
            elif method == 'undersample':
                sampler = RandomUnderSampler(random_state=self.random_state)
                
            else:
                logger.warning(f"طريقة غير معروفة: {method}, استخدام SMOTE")
                min_class_count = min(y.value_counts())
                k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1
                sampler = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
            
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            new_distribution = pd.Series(y_resampled).value_counts()
            logger.info(f"التوزيع الجديد: {new_distribution.to_dict()}")
            
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
            
        except Exception as e:
            logger.warning(f"فشل في معالجة عدم التوازن: {e}")
            logger.info("استخدام البيانات الأصلية")
            return X, y
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, 
                        experiment_name: str = "default",
                        test_size: float = 0.2) -> Dict[str, Any]:
        """
        تدريب جميع النماذج ومقارنتها
        
        Args:
            X: الميزات
            y: المتغير التابع
            experiment_name: اسم التجربة
            test_size: نسبة بيانات الاختبار
            
        Returns:
            نتائج جميع النماذج
        """
        
        logger.info(f"بدء تدريب جميع النماذج - تجربة: {experiment_name}")
        logger.info(f"شكل البيانات: {X.shape}")
        logger.info(f"توزيع الفئات: {y.value_counts().to_dict()}")
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, 
            random_state=self.random_state
        )
        
        logger.info(f"بيانات التدريب: {X_train.shape}, الاختبار: {X_test.shape}")
        
        # تدريب جميع النماذج
        models_config = self.get_models_config()
        results = {}
        
        for model_name in models_config.keys():
            try:
                result = self.train_single_model(
                    model_name, X_train, X_test, y_train, y_test
                )
                results[model_name] = result
                
                # تتبع أفضل نموذج
                if 'metrics' in result:
                    auc_score = result['metrics']['auc']
                    if auc_score > self.best_score:
                        self.best_score = auc_score
                        self.best_model = {
                            'name': model_name,
                            'model': result['model'],
                            'scaler': result['scaler'],
                            'auc': auc_score
                        }
                
            except Exception as e:
                logger.error(f"فشل تدريب {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # ملخص النتائج
        self.results = results
        
        logger.info("انتهى تدريب جميع النماذج")
        if self.best_model:
            logger.info(f"أفضل نموذج: {self.best_model['name']} (AUC = {self.best_model['auc']:.4f})")
        
        return results
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: pd.Series, y_test: pd.Series, 
                          balance_data: bool = True) -> Dict[str, Any]:
        """
        تدريب نموذج واحد
        
        Args:
            model_name: اسم النموذج
            X_train, X_test: بيانات التدريب والاختبار
            y_train, y_test: التسميات
            balance_data: معالجة عدم التوازن
            
        Returns:
            نتائج النموذج
        """
        
        logger.info(f"تدريب نموذج: {model_name}")
        
        models_config = self.get_models_config()
        
        if model_name not in models_config:
            raise ValueError(f"نموذج غير مدعوم: {model_name}")
        
        config = models_config[model_name]
        model = config['model']
        
        try:
            # تطبيق التطبيع إذا لزم الأمر
            if config['scale_features']:
                scaler = RobustScaler()
                X_train_processed = pd.DataFrame(
                    scaler.fit_transform(X_train), 
                    columns=X_train.columns,
                    index=X_train.index
                )
                X_test_processed = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )
                self.scalers[model_name] = scaler
            else:
                X_train_processed = X_train
                X_test_processed = X_test
                self.scalers[model_name] = None
            
            # معالجة عدم التوازن
            if balance_data:
                X_train_balanced, y_train_balanced = self.handle_imbalanced_data(
                    X_train_processed, y_train, method='smote'
                )
            else:
                X_train_balanced = X_train_processed
                y_train_balanced = y_train
            
            # التدريب
            model.fit(X_train_balanced, y_train_balanced)
            
            # التنبؤ
            y_pred = model.predict(X_test_processed)
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            
            # حساب المقاييس
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc': roc_auc_score(y_test, y_pred_proba),
                'avg_precision': average_precision_score(y_test, y_pred_proba)
            }
            
            # التحقق المتقاطع
            cv_scores = cross_val_score(
                model, X_train_balanced, y_train_balanced,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='roc_auc', n_jobs=-1
            )
            
            metrics.update({
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            })
            
            # أهمية الميزات
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X_train.columns, model.feature_importances_))
                # ترتيب حسب الأهمية
                feature_importance = dict(sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:20])  # أهم 20 ميزة
            
            # النتائج
            results = {
                'model': model,
                'scaler': self.scalers[model_name],
                'metrics': metrics,
                'feature_importance': feature_importance,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'description': config['description'],
                'test_auc': metrics['auc'],
                'test_precision': metrics['precision'],
                'test_recall': metrics['recall'],
                'test_f1': metrics['f1']
            }
            
            logger.info(f"{model_name} - AUC: {metrics['auc']:.4f}, "
                       f"Precision: {metrics['precision']:.4f}, "
                       f"Recall: {metrics['recall']:.4f}, "
                       f"F1: {metrics['f1']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"خطأ في تدريب {model_name}: {e}")
            return {'error': str(e), 'model_name': model_name}
    
    def save_best_model(self, save_dir: str):
        """حفظ أفضل نموذج"""
        
        if not self.best_model:
            logger.warning("لا يوجد نموذج أفضل لحفظه")
            return
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # حفظ النموذج
        model_path = save_dir / f"best_model_{self.best_model['name']}_{timestamp}.joblib"
        joblib.dump(self.best_model['model'], model_path)
        
        # حفظ المعالج إذا وُجد
        scaler_path = None
        if self.best_model['scaler'] is not None:
            scaler_path = save_dir.parent / "scalers" / f"scaler_{self.best_model['name']}_{timestamp}.joblib"
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.best_model['scaler'], scaler_path)
        
        # حفظ معلومات النموذج
        metadata = {
            'model_name': self.best_model['name'],
            'auc_score': self.best_model['auc'],
            'timestamp': timestamp,
            'model_path': str(model_path),
            'scaler_path': str(scaler_path) if scaler_path else None
        }
        
        metadata_path = save_dir.parent / f"model_metadata_{timestamp}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"تم حفظ أفضل نموذج: {model_path}")
        
        return {
            'model_path': model_path,
            'metadata_path': metadata_path,
            'scaler_path': scaler_path
        }