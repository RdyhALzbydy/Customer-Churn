#!/usr/bin/env python3
"""
سكريبت تقييم النماذج المدربة
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from radiya.models.evaluator import ModelEvaluator
from radiya.utils.logger import setup_logger
from radiya.utils.validators import ModelVisualizer
from radiya.config import *

import joblib
import pandas as pd
import json

def main():
    logger = setup_logger("evaluate_models", LOGS_DIR / "evaluation.log")
    logger.info("بدء تقييم النماذج")
    
    try:
        # البحث عن النماذج المحفوظة
        model_files = list((Path(MODEL_SAVE_PATH)).glob("*.joblib"))
        
        if not model_files:
            logger.error("لا توجد نماذج محفوظة للتقييم")
            return
        
        evaluator = ModelEvaluator()
        visualizer = ModelVisualizer()
        
        all_results = {}
        
        for model_file in model_files:
            logger.info(f"تقييم النموذج: {model_file.name}")
            
            # تحميل النموذج
            model = joblib.load(model_file)
            
            # تحميل بيانات الاختبار (يجب حفظها من مرحلة التدريب)
            test_data_file = PROCESSED_DATA_PATH / f"test_data_{model_file.stem}.csv"
            
            if test_data_file.exists():
                test_data = pd.read_csv(test_data_file)
                X_test = test_data.drop('churned', axis=1)
                y_test = test_data['churned']
                
                # التقييم
                results = evaluator.evaluate_model(model, X_test, y_test)
                all_results[model_file.stem] = results
                
                # إنشاء الرسوم البيانية
                visualizer.plot_model_performance(
                    y_test, 
                    model.predict_proba(X_test)[:, 1],
                    save_path=REPORTS_DIR / "figures" / f"{model_file.stem}_performance.html"
                )
                
                logger.info(f"AUC: {results['auc']:.4f}, Precision: {results['precision']:.4f}")
            else:
                logger.warning(f"بيانات الاختبار غير موجودة لـ {model_file.name}")
        
        # مقارنة النماذج
        if len(all_results) > 1:
            comparison = evaluator.compare_models(all_results)
            visualizer.plot_models_comparison(
                comparison,
                save_path=REPORTS_DIR / "figures" / "models_comparison.html"
            )
        
        # حفظ نتائج التقييم
        evaluation_file = REPORTS_DIR / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"تم حفظ نتائج التقييم في: {evaluation_file}")
        
        # طباعة النتائج
        print("\n" + "="*60)
        print("نتائج تقييم النماذج")
        print("="*60)
        
        for model_name, results in all_results.items():
            print(f"\n{model_name}:")
            print(f"  AUC Score: {results['auc']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  F1 Score: {results['f1']:.4f}")
            print(f"  Accuracy: {results['accuracy']:.4f}")
        
        if len(all_results) > 1:
            best_model = max(all_results.keys(), key=lambda k: all_results[k]['auc'])
            print(f"\nأفضل نموذج: {best_model} (AUC = {all_results[best_model]['auc']:.4f})")
        
    except Exception as e:
        logger.error(f"خطأ في التقييم: {e}")
        raise

if __name__ == "__main__":
    from datetime import datetime
    main()