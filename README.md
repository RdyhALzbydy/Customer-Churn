# 🎵 Customer Churn Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![API](https://img.shields.io/badge/API-FastAPI-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**نظام متكامل للتنبؤ بانسحاب العملاء من منصات البث الموسيقي**  
*Complete ML system for predicting customer churn in music streaming platforms*

[English](#english-version) | [العربية](#النسخة-العربية)

---

[![GitHub Stars](https://img.shields.io/github/stars/RdyhALzbydy/Customer-Churn?style=social)](https://github.com/RdyhALzbydy/Customer-Churn)
[![GitHub Forks](https://img.shields.io/github/forks/RdyhALzbydy/Customer-Churn?style=social)](https://github.com/RdyhALzbydy/Customer-Churn/fork)

</div>

---

## 🌟 النسخة العربية

### 📋 جدول المحتويات
- [نظرة عامة](#-نظرة-عامة)
- [المزايا الرئيسية](#-المزايا-الرئيسية)  
- [التقنيات المستخدمة](#-التقنيات-المستخدمة)
- [النتائج والأداء](#-النتائج-والأداء)
- [هيكل المشروع](#-هيكل-المشروع)
- [التثبيت والإعداد](#-التثبيت-والإعداد)
- [الاستخدام](#-الاستخدام)
- [واجهة برمجة التطبيقات](#-واجهة-برمجة-التطبيقات)
- [النشر](#-النشر)
- [المساهمة](#-المساهمة)
- [الترخيص](#-الترخيص)

---

### 🎯 نظرة عامة

**Customer Churn Prediction** هو نظام تعلم آلة متكامل مصمم للتنبؤ بانسحاب العملاء من منصات البث الموسيقي. يقدم المشروع حلولاً شاملة من تحليل البيانات إلى النشر والتطبيق العملي.

#### لماذا هذا المشروع؟
- 🎯 **دقة عالية**: نتائج تصل إلى 88% AUC مع نماذج محسّنة
- ⚡ **قابلية التوسع**: مصمم للتعامل مع البيانات الكبيرة
- 🔄 **مرونة كاملة**: من التدريب إلى النشر والتطبيق

- 🚀 **جاهز للإنتاج**: نشر سهل عبر Docker و docker-compose
- 🌐 **واجهات متعددة**: API + Web Interface للاستخدام المتنوع

---

<img width="1536" height="1024" alt="ChatGPT Image 26 أغسطس 2025، 05_50_04 م" src="https://github.com/user-attachments/assets/7d59b571-ed29-4d35-9cc5-50d555c45885" />
### ✨ المزايا الرئيسية

#### 📊 معالجة البيانات المتقدمة
- تحميل وتنظيف البيانات الخام
- هندسة الميزات الذكية
- معالجة البيانات المفقودة
- توازن الفئات باستخدام SMOTE

#### 🤖 نماذج التعلم الآلي
- **XGBoost**: الأداء الأفضل مع 88% AUC
- **LightGBM**: سرعة عالية وكفاءة
- **Random Forest**: استقرار وموثوقية
- **Logistic Regression**: بساطة وقابلية تفسير

#### 🌐 خدمات API شاملة
- تنبؤ فردي للعملاء
- تنبؤ جماعي للدفع
- رفع وتحليل ملفات البيانات
- تقارير مفصلة ومرئية

#### 💻 واجهة ويب تفاعلية
- صفحات بديهية للتحليل
- رفع الملفات بسهولة
- عرض النتائج والإحصائيات
- تصميم متجاوب

---

### 🛠 التقنيات المستخدمة

#### Machine Learning & Data Science
```
scikit-learn     # نماذج التعلم الآلي الأساسية
XGBoost         # تعزيز التدرج المتقدم
LightGBM        # تعلم آلة سريع وفعال
pandas          # معالجة البيانات
numpy           # الحوسبة العلمية
imbalanced-learn # معالجة عدم التوازن
```

#### Web Development & API
```
FastAPI         # إطار عمل API عالي الأداء
Uvicorn         # خادم ASGI
HTML/CSS/JS     # واجهة المستخدم
Jinja2          # قوالب HTML
```

#### Visualization & Analytics
```
matplotlib      # الرسوم البيانية الأساسية
seaborn         # تصور البيانات الإحصائي
plotly          # الرسوم التفاعلية
MLflow          # تتبع التجارب
```

#### Deployment & DevOps
```
Docker          # الحاويات
docker-compose  # إدارة الخدمات
pytest          # الاختبارات
black           # تنسيق الكود
ruff            # فحص الكود
```

---

### 📊 النتائج والأداء

#### مقاييس الأداء للنماذج

| النموذج | AUC Score | Precision | Recall | F1-Score | وقت التدريب |
|---------|-----------|-----------|---------|----------|-------------|
| **XGBoost** | **0.88** | **0.78** | **0.75** | **0.76** | 15s |
| LightGBM | 0.87 | 0.76 | 0.73 | 0.74 | 8s |
| Random Forest | 0.85 | 0.74 | 0.71 | 0.72 | 12s |
| Logistic Regression | 0.82 | 0.71 | 0.68 | 0.69 | 3s |

#### ميزات النموذج الرائدة
1. **الاستخدام اليومي** - أقوى مؤشر للانسحاب
2. **عدد الجلسات** - نشاط المستخدم
3. **نوع العضوية** - Free vs Premium
4. **التفاعل مع الإعلانات** - سلوك المستخدم
5. **مدة الاشتراك** - ولاء العميل

---

### 📁 هيكل المشروع

```
customer-churn/
├── 🚀 run_radiya.py           # نقطة البداية الرئيسية
├── 📋 requirements.txt        # المتطلبات
├── ⚙️ setup.py               # إعداد الحزمة
├── 🐳 docker-compose.yml     # خدمات Docker
├── 📄 Dockerfile             # صورة Docker
├── 🛠 Makefile               # أوامر التشغيل
├── 🔧 .env                   # متغيرات البيئة
│
├── 📊 data/
│   ├── raw/                  # البيانات الخام
│   ├── processed/            # البيانات المعالجة
│   └── uploads/              # الملفات المرفوعة
│
├── 💻 src/radiya/
│   ├── config.py            # إعدادات المشروع
│   ├── data/                # معالجة البيانات
│   │   ├── loader.py        # تحميل البيانات
│   │   ├── preprocessor.py  # معالجة أولية
│   │   └── feature_engineer.py # هندسة الميزات
│   ├── models/              # نماذج التعلم الآلي
│   │   ├── trainer.py       # تدريب النماذج
│   │   ├── predictor.py     # التنبؤ
│   │   └── evaluator.py     # التقييم
│   ├── api/                 # واجهة برمجة التطبيقات
│   │   ├── main.py          # تطبيق FastAPI الرئيسي
│   │   ├── routes/          # مسارات API
│   │   ├── schemas.py       # مخططات البيانات
│   │   └── utils.py         # أدوات مساعدة
│   ├── web/                 # واجهة الويب
│   │   ├── templates/       # قوالب HTML
│   │   └── static/          # CSS, JS, Assets
│   └── utils/               # أدوات عامة
│       ├── logger.py        # نظام السجلات
│       ├── file_handler.py  # معالجة الملفات
│       └── validators.py    # التحقق من البيانات
│
├── 🎯 models/               # النماذج المحفوظة
├── 📈 reports/              # التقارير والنتائج
├── 📝 logs/                 # ملفات السجل
└── 🧪 tests/                # الاختبارات
```

---

### 🚀 التثبيت والإعداد

#### التثبيت التقليدي

```bash
# استنساخ المشروع
git clone https://github.com/RdyhALzbydy/Customer-Churn.git
cd Customer-Churn

# إنشاء بيئة افتراضية
python -m venv venv

# تفعيل البيئة الافتراضية
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# تثبيت المتطلبات
pip install -r requirements.txt

# تشغيل المشروع
python run_radiya.py
```

#### التثبيت باستخدام Docker (موصى به)

```bash
# استنساخ المشروع
git clone https://github.com/RdyhALzbydy/Customer-Churn.git
cd Customer-Churn

# تشغيل جميع الخدمات
docker-compose up --build

# الوصول للخدمات:
# 🌐 API & Web Interface: http://localhost:8000
# 📊 MLflow Tracking: http://localhost:5000
```

#### التثبيت السريع للتطوير

```bash
# باستخدام Makefile
make install     # تثبيت المتطلبات
make run         # تشغيل المشروع
make test        # تشغيل الاختبارات
make docker      # تشغيل Docker
```

---

### 📖 الاستخدام

#### 1. التدريب والتحليل الأساسي

```bash
# تشغيل المشروع كاملاً (تدريب + تقييم + حفظ النماذج)
python run_radiya.py

# أو باستخدام معاملات مخصصة
python run_radiya.py --model xgboost --test-size 0.2 --cv-folds 5
```

#### 2. تشغيل خدمة API

```bash
# تشغيل API للتنبؤ والتطبيقات الخارجية
uvicorn src.radiya.api.main:app --reload --port 8000

# مع إعدادات متقدمة
uvicorn src.radiya.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### 3. استخدام واجهة الويب

1. افتح المتصفح واذهب إلى: `http://localhost:8000`
2. اختر إما "تحليل فردي" أو "تحليل ملف"
3. أدخل البيانات أو ارفع ملف CSV
4. شاهد النتائج والتصورات

#### 4. استخدام MLflow لتتبع التجارب

```bash
# تشغيل MLflow UI
mlflow ui --port 5000

# عرض التجارب على: http://localhost:5000
```

---

### 🔌 واجهة برمجة التطبيقات

#### نقاط النهاية الأساسية

##### تنبؤ فردي
```http
POST /predict
Content-Type: application/json

{
  "userId": "12345",
  "gender": "M",
  "level": "paid",
  "total_sessions": 150,
  "total_songs": 3000,
  "avg_songs_per_session": 20,
  "days_since_registration": 90
}
```

##### تنبؤ جماعي
```http
POST /predict/batch
Content-Type: multipart/form-data

file: users_data.csv
```

##### رفع وتحليل البيانات
```http
POST /upload
Content-Type: multipart/form-data

file: churn_data.csv
```

##### الحصول على تحليل سابق
```http
GET /analysis/{analysis_id}
```

#### أمثلة للاستخدام بـ Python

```python
import requests
import json

# تنبؤ فردي
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "userId": "user123",
        "gender": "F", 
        "level": "free",
        "total_sessions": 50,
        "total_songs": 800,
        "avg_songs_per_session": 16,
        "days_since_registration": 30
    }
)
result = response.json()
print(f"احتمالية الانسحاب: {result['churn_probability']:.2%}")
```

#### التوثيق التفاعلي

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

### 🐳 النشر

#### البيئة التطويرية

```bash
# تشغيل خدمة API فقط
docker-compose up radiya-api

# تشغيل جميع الخدمات (API + MLflow)
docker-compose up
```

#### البيئة الإنتاجية

```bash
# إنتاج مع إعدادات محسّنة
docker-compose -f docker-compose.prod.yml up -d

# مع Load Balancer
docker-compose -f docker-compose.prod.yml --scale radiya-api=3 up -d
```

#### النشر على السحابة

##### AWS EC2
```bash
# رفع على AWS
scp -r . ec2-user@your-instance:/home/ec2-user/customer-churn/
ssh ec2-user@your-instance
cd customer-churn && docker-compose up -d
```

##### Google Cloud Platform
```bash
# النشر على GCP
gcloud run deploy customer-churn \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

### 🧪 الاختبارات

```bash
# تشغيل جميع الاختبارات
pytest tests/

# اختبارات API فقط
pytest tests/test_api.py -v

# اختبارات مع تغطية الكود
pytest tests/ --cov=src --cov-report=html

# اختبارات الأداء
pytest tests/test_performance.py --benchmark-only
```

---

### 🤝 المساهمة

نرحب بمساهماتكم! يرجى اتباع الخطوات التالية:

1. **Fork المشروع**
   ```bash
   git clone https://github.com/RdyhALzbydy/Customer-Churn.git
   ```

2. **إنشاء فرع للميزة الجديدة**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **كتابة وتطبيق التغييرات**
   ```bash
   git commit -m "Add amazing feature"
   ```

4. **رفع التغييرات**
   ```bash
   git push origin feature/amazing-feature
   ```

5. **فتح Pull Request**

#### دليل المساهمة
- اتبع معايير كتابة الكود (Black, Ruff)
- أضف اختبارات للميزات الجديدة
- حدّث التوثيق عند الحاجة
- اكتب رسائل commit واضحة

---

### 📄 الترخيص

هذا المشروع مرخّص تحت [MIT License](LICENSE). يمكنك استخدامه، تعديله، وتوزيعه بحرية.

---

### 📞 التواصل والدعم

- **👨‍💻 المطور**: Eng. Radiya
- **📧 البريد الإلكتروني**: [rdyhalzbydy@gmail.com](mailto:rdyhalzbydy@gmail.com)
- **🔗 GitHub**: [Customer Churn Project](https://github.com/RdyhALzbydy/Customer-Churn)
- **💼 LinkedIn**: [تواصل معي](https://linkedin.com/in/your-profile)

#### الحصول على المساعدة
- 🐛 **Issues**: [إبلاغ عن مشكلة](https://github.com/RdyhALzbydy/Customer-Churn/issues)
- 💡 **Discussions**: [مناقشات المجتمع](https://github.com/RdyhALzbydy/Customer-Churn/discussions)
- 📖 **Wiki**: [دليل شامل](https://github.com/RdyhALzbydy/Customer-Churn/wiki)

---

## English Version

### 🎯 Overview

**Customer Churn Prediction** is a comprehensive machine learning system designed to predict customer churn in music streaming platforms. The project provides end-to-end solutions from data analysis to deployment and practical application.

### ✨ Key Features

- 🔄 **Advanced Data Processing**: Loading, cleaning, feature engineering
- 🤖 **Multiple ML Models**: XGBoost, LightGBM, Random Forest, Logistic Regression  
- 🌐 **Complete API Services**: Individual/batch prediction, file upload, analysis
- 💻 **Interactive Web Interface**: User-friendly pages for analysis and results
- 📊 **Comprehensive Reports**: Detailed metrics and visualizations
- 🐳 **Production Ready**: Easy deployment with Docker and docker-compose

### 🚀 Quick Start

```bash
git clone https://github.com/RdyhALzbydy/Customer-Churn.git
cd Customer-Churn
pip install -r requirements.txt
python run_radiya.py
```

### 🐳 Docker Quick Start

```bash
docker-compose up --build
# API & Web: http://localhost:8000
# MLflow: http://localhost:5000
```

### 📊 Performance Results

| Model | AUC Score | Precision | Recall | F1-Score |
|-------|-----------|-----------|---------|----------|
| **XGBoost** | **0.88** | **0.78** | **0.75** | **0.76** |
| LightGBM | 0.87 | 0.76 | 0.73 | 0.74 |
| Random Forest | 0.85 | 0.74 | 0.71 | 0.72 |
| Logistic Regression | 0.82 | 0.71 | 0.68 | 0.69 |

### 📚 Documentation

- **API Docs**: `http://localhost:8000/docs`
- **Project Wiki**: [GitHub Wiki](https://github.com/RdyhALzbydy/Customer-Churn/wiki)
- **Issues**: [Report Bugs](https://github.com/RdyhALzbydy/Customer-Churn/issues)

---

<div align="center">

⭐ **إذا أعجبك هذا المشروع، لا تنس إعطاءه نجمة!**  
⭐ **If you like this project, please give it a star!**

[![GitHub Stars](https://img.shields.io/github/stars/RdyhALzbydy/Customer-Churn?style=social)](https://github.com/RdyhALzbydy/Customer-Churn)

**Made with ❤️ by Eng. Radiya**

</div>
