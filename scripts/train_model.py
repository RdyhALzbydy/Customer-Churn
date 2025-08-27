#!/usr/bin/env python3
"""
سكريبت تدريب النماذج الرئيسي
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from radiya.data.loader import DataLoader
from radiya.data.feature_engineer import FeatureEngineer
from radiya.models.trainer import ModelTrainer
from radiya.utils.logger import setup_logger
from radiya.config import *

import mlflow

def main():
    # إعداد السجلات
    logger = setup_logger("train_model", LOGS_DIR / "training.log")
    logger.info("بدء تدريب نماذج رضية")
    
    # إعداد MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    try:
        # تحميل البيانات
        logger.info("تحميل البيانات...")
        loader = DataLoader(DATA_PATH)
        df = loader.load_data()
        
        # هندسة الميزات
        logger.info("هندسة الميزات...")
        engineer = FeatureEngineer()
        features_df = engineer.create_features(df)
        
        # تجريب طرق تعريف الانسحاب المختلفة
        results = {}
        
        for churn_method in CHURN_METHODS:
            logger.info(f"تدريب النماذج باستخدام طريقة: {churn_method}")
            
            # تعريف الانسحاب
            churn_labels = engineer.define_churn(df, method=churn_method)
            
            # دمج البيانات
            ml_data = features_df.merge(churn_labels, on='userId')
            X = ml_data.drop(['userId', 'churned'], axis=1)
            y = ml_data['churned']
            
            # التدريب
            trainer = ModelTrainer()
            method_results = trainer.train_all_models(X, y, churn_method)
            results[churn_method] = method_results
        
        # أفضل نموذج عام
        best_method = max(results.keys(), key=lambda k: max([
            result.get('test_auc', 0) for result in results[k].values() 
            if isinstance(result, dict)
        ]))
        
        best_model_name = max(results[best_method].keys(), key=lambda k: 
            results[best_method][k].get('test_auc', 0) if isinstance(results[best_method][k], dict) else 0
        )
        
        logger.info(f"أفضل نموذج: {best_model_name} بطريقة {best_method}")
        logger.info(f"AUC Score: {results[best_method][best_model_name]['test_auc']:.4f}")
        
        # حفظ النتائج
        import json
        results_file = REPORTS_DIR / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # تحويل النتائج لتكون قابلة للتسلسل
        serializable_results = {}
        for method, method_results in results.items():
            serializable_results[method] = {}
            for model_name, result in method_results.items():
                if isinstance(result, dict):
                    serializable_results[method][model_name] = {
                        k: v for k, v in result.items() 
                        if k in ['test_auc', 'test_precision', 'test_recall', 'test_f1', 'cv_mean', 'cv_std']
                    }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"تم حفظ النتائج في: {results_file}")
        
    except Exception as e:
        logger.error(f"خطأ في التدريب: {e}")
        raise

if __name__ == "__main__":
    from datetime import datetime
    main()