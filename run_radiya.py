#!/usr/bin/env python3
"""
سكريبت التشغيل الكامل لمشروع رضية
تحليل وتدريب ومقارنة نماذج التنبؤ بالانسحاب
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# إضافة مسار المشروع
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

# استيراد مكونات المشروع
try:
    from radiya.data.loader import DataLoader
    from radiya.data.feature_engineer import FeatureEngineer
    from radiya.models.trainer import ModelTrainer
except ImportError as e:
    print(f"خطأ في استيراد المكونات: {e}")
    print("تأكد من وجود ملفات المشروع في المسارات الصحيحة")
    sys.exit(1)

# إعداد MLflow
try:
    import mlflow
    mlflow.set_tracking_uri("sqlite:///radiya_experiments.db")
    mlflow.set_experiment("radiya_churn_prediction")
except ImportError:
    print("تحذير: MLflow غير متوفر، لن يتم تسجيل التجارب")
    mlflow = None

# إعداد نظام السجلات
def setup_logging():
    """إعداد نظام السجلات"""
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / f"radiya_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("radiya_main")

def create_directories():
    """إنشاء المجلدات المطلوبة"""
    directories = [
        "data/raw", "data/processed", "models/saved_models", 
        "models/scalers", "reports/metrics", "reports/figures", "logs"
    ]
    
    for dir_path in directories:
        (project_root / dir_path).mkdir(parents=True, exist_ok=True)

def find_data_file():
    """البحث عن ملف البيانات"""
    possible_paths = [
        project_root / "data" / "raw" / "customer_churn_mini.json",
        project_root / "customer_churn_mini.json",
        Path("customer_churn_mini.json"),
        Path("data.json")
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None

def print_header():
    """طباعة رأس المشروع"""
    print("\n" + "="*80)
    print("                    مشروع رضية - Customer Churn Prediction")
    print("                         التنبؤ بانسحاب المستخدمين")
    print("="*80)
    print(f"وقت البدء: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

def run_data_analysis(data_path: str, logger):
    """تشغيل تحليل البيانات"""
    logger.info("المرحلة 1: تحليل البيانات")
    print("\n📊 المرحلة 1: تحليل البيانات")
    print("-" * 50)
    
    # تحميل البيانات
    loader = DataLoader(data_path)
    df = loader.load_data()
    
    # عرض ملخص البيانات
    summary = loader.get_data_summary()
    
    print(f"إجمالي المستخدمين: {summary['basic_stats']['unique_users']:,}")
    print(f"إجمالي الأحداث: {summary['basic_stats']['clean_records']:,}")
    print(f"إجمالي الأغاني المشغلة: {summary['basic_stats']['total_songs']:,}")
    print(f"الفترة الزمنية: {summary['basic_stats']['date_range'][0].strftime('%Y-%m-%d')} إلى {summary['basic_stats']['date_range'][1].strftime('%Y-%m-%d')}")
    
    if 'subscription_levels' in summary:
        print("\nتوزيع مستويات الاشتراك:")
        for level, count in summary['subscription_levels'].items():
            print(f"  {level}: {count:,} مستخدم ({count/summary['basic_stats']['unique_users']*100:.1f}%)")
    
    if 'page_distribution' in summary:
        print("\nأهم 5 صفحات:")
        top_pages = sorted(summary['page_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]
        for page, count in top_pages:
            print(f"  {page}: {count:,}")
    
    return df, loader

def run_feature_engineering(df: pd.DataFrame, logger):
    """تشغيل هندسة الميزات"""
    logger.info("المرحلة 2: هندسة الميزات")
    print("\n🔧 المرحلة 2: هندسة الميزات")
    print("-" * 50)
    
    engineer = FeatureEngineer()
    
    # إنشاء الميزات
    print("إنشاء ميزات المستخدمين...")
    features_df = engineer.create_features(df)
    
    print(f"تم إنشاء {len(features_df.columns)-1} ميزة لـ {len(features_df)} مستخدم")
    
    # تجريب طرق تعريف الانسحاب المختلفة
    churn_methods = ['cancellation', 'downgrade', 'combined', 'inactivity']
    churn_results = {}
    
    print("\nطرق تعريف الانسحاب:")
    for method in churn_methods:
        try:
            churn_labels = engineer.define_churn(df, method=method)
            churn_rate = churn_labels['churned'].mean() * 100
            churn_count = churn_labels['churned'].sum()
            
            print(f"  {method}: {churn_count:,} منسحب ({churn_rate:.1f}%)")
            churn_results[method] = {
                'labels': churn_labels,
                'rate': churn_rate,
                'count': churn_count
            }
        except Exception as e:
            print(f"  {method}: خطأ - {e}")
            churn_results[method] = None
    
    return features_df, engineer, churn_results

def run_model_training(features_df: pd.DataFrame, churn_results: dict, logger):
    """تشغيل تدريب النماذج"""
    logger.info("المرحلة 3: تدريب النماذج")
    print("\n🤖 المرحلة 3: تدريب النماذج")
    print("-" * 50)
    
    all_results = {}
    best_overall_model = None
    best_overall_score = 0
    
    # تدريب النماذج لكل طريقة تعريف انسحاب
    for method_name, churn_data in churn_results.items():
        if churn_data is None:
            continue
            
        print(f"\nتدريب النماذج باستخدام طريقة: {method_name}")
        print(f"معدل الانسحاب: {churn_data['rate']:.1f}%")
        
        # تحضير البيانات
        ml_data = features_df.merge(churn_data['labels'], on='userId')
        X = ml_data.drop(['userId', 'churned'], axis=1)
        y = ml_data['churned']
        
        if len(y.unique()) < 2:
            print(f"تخطي {method_name} - لا يوجد تنوع في الفئات")
            continue
        
        # تدريب النماذج
        trainer = ModelTrainer(random_state=42)
        method_results = trainer.train_all_models(X, y, experiment_name=method_name)
        
        # حفظ النتائج
        all_results[method_name] = {
            'trainer': trainer,
            'results': method_results,
            'churn_rate': churn_data['rate'],
            'data_shape': X.shape
        }
        
        # تتبع أفضل نموذج عام
        if trainer.best_model and trainer.best_score > best_overall_score:
            best_overall_score = trainer.best_score
            best_overall_model = {
                'method': method_name,
                'name': trainer.best_model['name'],
                'score': trainer.best_score,
                'trainer': trainer
            }
    
    return all_results, best_overall_model

def display_results(all_results: dict, best_overall_model: dict, logger):
    """عرض النتائج النهائية"""
    logger.info("المرحلة 4: عرض النتائج")
    print("\n📈 المرحلة 4: ملخص النتائج")
    print("=" * 80)
    
    # ملخص شامل لجميع النتائج
    summary_data = []
    
    for method_name, method_data in all_results.items():
        trainer = method_data['trainer']
        results = method_data['results']
        
        print(f"\n🎯 طريقة: {method_name.upper()}")
        print(f"معدل الانسحاب: {method_data['churn_rate']:.1f}%")
        print(f"شكل البيانات: {method_data['data_shape']}")
        print("-" * 60)
        
        # عرض نتائج كل نموذج
        for model_name, result in results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"{model_name:15} | AUC: {metrics['auc']:.4f} | "
                      f"Precision: {metrics['precision']:.4f} | "
                      f"Recall: {metrics['recall']:.4f} | "
                      f"F1: {metrics['f1']:.4f}")
                
                summary_data.append({
                    'Method': method_name,
                    'Model': model_name,
                    'AUC': metrics['auc'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1': metrics['f1'],
                    'CV_Mean': metrics['cv_mean'],
                    'CV_Std': metrics['cv_std']
                })
            else:
                print(f"{model_name:15} | خطأ: {result.get('error', 'غير محدد')}")
    
    # أفضل نموذج عام
    print("\n🏆 أفضل نموذج عام:")
    print("=" * 60)
    if best_overall_model:
        print(f"الطريقة: {best_overall_model['method']}")
        print(f"النموذج: {best_overall_model['name']}")
        print(f"AUC Score: {best_overall_model['score']:.4f}")
        
        # حفظ أفضل نموذج
        save_paths = best_overall_model['trainer'].save_best_model(
            project_root / "models" / "saved_models"
        )
        print(f"تم حفظ النموذج في: {save_paths['model_path']}")
    else:
        print("لم يتم العثور على نموذج صالح")
    
    # حفظ ملخص النتائج
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('AUC', ascending=False)
        
        results_file = project_root / "reports" / "metrics" / f"results_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(results_file, index=False)
        print(f"\nتم حفظ ملخص النتائج: {results_file}")
        
        # عرض أفضل 10 نماذج
        print("\n📊 أفضل 10 نماذج:")
        print("-" * 80)
        top_models = summary_df.head(10)
        for _, row in top_models.iterrows():
            print(f"{row['Method']:12} | {row['Model']:15} | AUC: {row['AUC']:.4f}")

def save_experiment_log(all_results: dict, best_overall_model: dict, data_path: str):
    """حفظ سجل التجربة"""
    
    experiment_log = {
        'timestamp': datetime.now().isoformat(),
        'data_path': data_path,
        'data_file_size_mb': Path(data_path).stat().st_size / 1024 / 1024,
        'total_methods_tested': len(all_results),
        'total_models_trained': sum(len(method_data['results']) for method_data in all_results.values()),
        'best_overall_model': {
            'method': best_overall_model['method'] if best_overall_model else None,
            'name': best_overall_model['name'] if best_overall_model else None,
            'score': best_overall_model['score'] if best_overall_model else 0
        },
        'method_results': {}
    }
    
    # تفاصيل كل طريقة
    for method_name, method_data in all_results.items():
        successful_models = [
            {
                'name': model_name,
                'auc': result['metrics']['auc'],
                'precision': result['metrics']['precision'],
                'recall': result['metrics']['recall'],
                'f1': result['metrics']['f1']
            }
            for model_name, result in method_data['results'].items() 
            if 'metrics' in result
        ]
        
        experiment_log['method_results'][method_name] = {
            'churn_rate': method_data['churn_rate'],
            'data_shape': method_data['data_shape'],
            'successful_models': len(successful_models),
            'best_model': max(successful_models, key=lambda x: x['auc']) if successful_models else None
        }
    
    # حفظ السجل
    log_file = project_root / "reports" / "metrics" / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_log, f, ensure_ascii=False, indent=2)
    
    print(f"تم حفظ سجل التجربة: {log_file}")

def main():
    """الوظيفة الرئيسية"""
    
    # إعداد أولي
    print_header()
    logger = setup_logging()
    create_directories()
    
    logger.info("بدء تشغيل مشروع رضية")
    
    try:
        # البحث عن ملف البيانات
        data_path = find_data_file()
        if not data_path:
            print("❌ لم يتم العثور على ملف البيانات!")
            print("\nضع ملف البيانات في أحد المواقع التالية:")
            print("  - data/raw/customer_churn_mini.json")
            print("  - customer_churn_mini.json")
            return
        
        print(f"✅ تم العثور على ملف البيانات: {data_path}")
        
        # تشغيل المراحل
        df, loader = run_data_analysis(data_path, logger)
        features_df, engineer, churn_results = run_feature_engineering(df, logger)
        all_results, best_overall_model = run_model_training(features_df, churn_results, logger)
        
        if all_results:
            display_results(all_results, best_overall_model, logger)
            save_experiment_log(all_results, best_overall_model, data_path)
        else:
            print("❌ لم يتم تدريب أي نماذج بنجاح")
        
        print("\n" + "="*80)
        print("✅ انتهى تشغيل مشروع رضية بنجاح!")
        print("="*80)
        
        logger.info("انتهى تشغيل المشروع بنجاح")
        
    except Exception as e:
        logger.error(f"خطأ في تشغيل المشروع: {e}")
        print(f"❌ خطأ في تشغيل المشروع: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()