"""
مدرب النماذج لمشروع رضية
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

# رسم بياني
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek

# معالجة عدم التوازن
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# النماذج
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

logger = logging.getLogger(__name__)

# MLflow للتتبع
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow غير متوفر، لن يتم تسجيل التجارب")

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

    def get_models_config(self) -> dict[str, dict]:
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
                              method: str = 'smote') -> tuple[pd.DataFrame, pd.Series]:
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
                        test_size: float = 0.2) -> dict[str, Any]:
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

        # إعداد MLflow experiment
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment(f"radiya_churn_{experiment_name}")
                logger.info(f"تم إعداد MLflow experiment: radiya_churn_{experiment_name}")
            except Exception as e:
                logger.warning(f"فشل إعداد MLflow: {e}")

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
                          balance_data: bool = True) -> dict[str, Any]:
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

            # تسجيل في MLflow
            if MLFLOW_AVAILABLE:
                try:
                    with mlflow.start_run(run_name=f"{model_name}"):
                        # تسجيل المعايير
                        clean_metrics = {}
                        for key, value in metrics.items():
                            try:
                                clean_metrics[key] = float(value) if np.isscalar(value) else float(np.mean(value))
                            except (ValueError, TypeError):
                                pass
                        mlflow.log_metrics(clean_metrics)

                        # تسجيل المعاملات
                        mlflow.log_params({
                            'model_type': model_name,
                            'description': config['description'],
                            'scale_features': config['scale_features'],
                            'train_size': len(X_train),
                            'test_size': len(X_test),
                            'n_features': X_train.shape[1]
                        })

                        # حفظ النموذج
                        mlflow.sklearn.log_model(model, "model")

                        # تسجيل ميزات مهمة إضافية
                        if feature_importance:
                            for feat, imp in list(feature_importance.items())[:5]:  # أهم 5 ميزات
                                try:
                                    # تحويل إلى float سواء كان scalar أو array
                                    imp_value = float(imp) if np.isscalar(imp) else float(np.mean(imp))
                                    mlflow.log_metric(f"importance_{feat}", imp_value)
                                except (ValueError, TypeError):
                                    pass  # تجاهل إذا فشل التحويل

                        # إنشاء الرسوم البيانية (معطل مؤقتاً لحل مشاكل matplotlib)
                        # self._create_and_log_plots(model_name, y_test, y_pred, y_pred_proba,
                        #                          feature_importance, confusion_matrix(y_test, y_pred))

                        logger.info(f"تم تسجيل {model_name} في MLflow")

                except Exception as e:
                    logger.warning(f"فشل تسجيل {model_name} في MLflow: {e}")

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

    def _create_and_log_plots(self, model_name: str, y_test, y_pred, y_pred_proba,
                             feature_importance, conf_matrix):
        """إنشاء الرسوم البيانية وتسجيلها في MLflow"""

        try:
            # إنشاء مجلد مؤقت للرسوم
            with tempfile.TemporaryDirectory() as tmpdir:

                # 1. Confusion Matrix
                plt.figure(figsize=(8, 6))
                if SEABORN_AVAILABLE:
                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                else:
                    plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
                    plt.colorbar()
                    for i in range(conf_matrix.shape[0]):
                        for j in range(conf_matrix.shape[1]):
                            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center')
                plt.title(f'{model_name} - Confusion Matrix')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                confusion_path = os.path.join(tmpdir, 'confusion_matrix.png')
                plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(confusion_path, "plots")
                plt.close()

                # 2. Feature Importance (إذا كانت متوفرة)
                if feature_importance and len(feature_importance) > 0:
                    plt.figure(figsize=(10, 6))
                    features = list(feature_importance.keys())[:10]  # أهم 10 ميزات
                    importances = [float(feature_importance[f]) for f in features]

                    if SEABORN_AVAILABLE:
                        sns.barplot(x=importances, y=features)
                    else:
                        plt.barh(range(len(features)), importances)
                        plt.yticks(range(len(features)), features)
                    plt.title(f'{model_name} - Top 10 Feature Importance')
                    plt.xlabel('Importance')
                    importance_path = os.path.join(tmpdir, 'feature_importance.png')
                    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
                    mlflow.log_artifact(importance_path, "plots")
                    plt.close()

                # 3. ROC Curve (إذا كانت الاحتماليات متوفرة)
                if y_pred_proba is not None:
                    from sklearn.metrics import auc, roc_curve
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)

                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2,
                            label=f'ROC curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'{model_name} - ROC Curve')
                    plt.legend(loc="lower right")
                    roc_path = os.path.join(tmpdir, 'roc_curve.png')
                    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
                    mlflow.log_artifact(roc_path, "plots")
                    plt.close()

        except Exception as e:
            logger.warning(f"فشل في إنشاء الرسوم البيانية لـ {model_name}: {e}")
