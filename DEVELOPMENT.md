# دليل المطورين - مشروع رضية

## نظرة عامة
دليل شامل للمطورين الذين يريدون المساهمة في مشروع رضية أو فهم كيفية عمل النظام وتطويره.

---

## 1. إعداد بيئة التطوير

### المتطلبات الأساسية
- **Python 3.9+**: إصدار Python المطلوب
- **Git**: لإدارة الإصدارات
- **Docker**: للحاويات (اختياري)
- **uv**: مدير الحزم السريع (موصى به)

### إعداد البيئة المحلية
```bash
# استنساخ المشروع
git clone https://github.com/your-repo/radiya-project.git
cd radiya-project

# إنشاء البيئة الافتراضية
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# أو .venv\Scripts\activate  # Windows

# تثبيت التبعيات
pip install -e ".[dev]"

# أو باستخدام uv (أسرع)
uv sync --dev
```

### إعداد pre-commit hooks
```bash
# تثبيت pre-commit hooks
pre-commit install

# اختبار الإعداد
pre-commit run --all-files
```

---

## 2. هيكل المشروع

### نظرة على الهيكل
```
radiya-project/
├── src/radiya/              # كود المشروع الرئيسي
│   ├── __init__.py
│   ├── api/                 # FastAPI application
│   │   ├── main.py         # نقطة دخول التطبيق
│   │   └── routes/         # مسارات API
│   ├── features/           # هندسة الميزات
│   │   └── engineer.py
│   ├── models/             # نماذج التعلم الآلي
│   │   └── trainer.py
│   ├── utils/              # أدوات مساعدة
│   │   └── validators.py
│   ├── web/                # الواجهة الأمامية
│   └── scheduler.py        # نظام إعادة التدريب
├── tests/                  # اختبارات الوحدة
├── data/                   # البيانات
│   ├── raw/               # بيانات خام
│   └── processed/         # بيانات معالجة
├── models/                 # النماذج المحفوظة
├── reports/               # التقارير والنتائج
├── docs/                  # التوثيق
├── pyproject.toml         # إعدادات المشروع
├── docker-compose.yml     # إعدادات Docker
└── README.md
```

### المكونات الأساسية

#### أ. واجهة برمجة التطبيقات (API Layer)
```python
# src/radiya/api/main.py
from fastapi import FastAPI
from .routes import prediction, analysis, upload, scheduler

app = FastAPI(title="Radiya API")
app.include_router(prediction.router, prefix="/api/v1")
# ...
```

#### ب. هندسة الميزات (Feature Engineering)
```python
# src/radiya/features/engineer.py
class SimpleFeatureEngineer:
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """استخراج الميزات من البيانات الخام"""
        # منطق هندسة الميزات
        pass
```

#### ج. تدريب النماذج (Model Training)
```python
# src/radiya/models/trainer.py
class ModelTrainer:
    def train_all_models(self, X, y) -> Dict[str, Any]:
        """تدريب جميع النماذج ومقارنتها"""
        # منطق التدريب
        pass
```

---

## 3. معايير الكود

### Python Code Style
نستخدم الأدوات التالية لضمان جودة الكود:
- **ruff**: لفحص الكود وإصلاح المشاكل
- **black**: لتنسيق الكود
- **isort**: لترتيب imports
- **mypy**: للتحقق من الأنواع

```bash
# تشغيل جميع فحوصات الجودة
ruff check src/
black --check src/
isort --check-only src/
mypy src/
```

### Type Hints
استخدم Type Hints في جميع الوظائف:
```python
from typing import Dict, List, Optional, Union
import pandas as pd

def process_data(
    df: pd.DataFrame, 
    method: str = "default"
) -> Dict[str, Union[float, int]]:
    """معالج البيانات مع Type Hints"""
    return {"processed": len(df)}
```

### Docstrings
استخدم docstrings واضحة:
```python
def calculate_churn_rate(df: pd.DataFrame, method: str) -> float:
    """
    حساب معدل الانسحاب
    
    Args:
        df: إطار البيانات الذي يحتوي على بيانات العملاء
        method: طريقة حساب الانسحاب ('cancellation', 'downgrade', etc.)
        
    Returns:
        معدل الانسحاب كنسبة مئوية
        
    Raises:
        ValueError: إذا كانت الطريقة غير مدعومة
        
    Example:
        >>> df = pd.read_json('data.json')
        >>> rate = calculate_churn_rate(df, 'cancellation')
        >>> print(f"معدل الانسحاب: {rate:.2f}%")
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"طريقة غير مدعومة: {method}")
    
    # منطق الحساب
    return rate
```

---

## 4. الاختبارات

### بنية الاختبارات
```
tests/
├── conftest.py              # إعدادات pytest
├── unit/                    # اختبارات الوحدة
│   ├── test_features.py
│   ├── test_models.py
│   └── test_utils.py
├── integration/             # اختبارات التكامل
│   ├── test_api.py
│   └── test_pipeline.py
└── fixtures/               # بيانات اختبار
    └── sample_data.json
```

### كتابة الاختبارات
```python
# tests/unit/test_features.py
import pytest
import pandas as pd
from src.radiya.features.engineer import SimpleFeatureEngineer

class TestSimpleFeatureEngineer:
    @pytest.fixture
    def sample_data(self):
        """بيانات تجريبية للاختبار"""
        return pd.DataFrame({
            'userId': ['user1', 'user2'],
            'sessionId': [1, 2],
            'page': ['NextSong', 'Home'],
            'ts': [1640995200000, 1640995260000]
        })
    
    @pytest.fixture
    def engineer(self):
        return SimpleFeatureEngineer()
    
    def test_engineer_features(self, engineer, sample_data):
        """اختبار استخراج الميزات"""
        features = engineer.engineer_features(sample_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert 'userId' in features.columns
        
    def test_define_churn_cancellation(self, engineer, sample_data):
        """اختبار تعريف الانسحاب - الإلغاء"""
        churn_labels = engineer.define_churn(
            sample_data, method='cancellation'
        )
        
        assert 'churned' in churn_labels.columns
        assert churn_labels['churned'].dtype == bool
```

### تشغيل الاختبارات
```bash
# تشغيل جميع الاختبارات
pytest

# تشغيل اختبارات محددة
pytest tests/unit/test_features.py

# تشغيل مع تغطية الكود
pytest --cov=src/radiya --cov-report=html

# تشغيل اختبارات متوازية
pytest -n auto
```

---

## 5. إضافة ميزات جديدة

### سير العمل (Workflow)
1. **إنشاء فرع جديد**
   ```bash
   git checkout -b feature/new-feature-name
   ```

2. **كتابة الكود**
   - اتبع معايير الكود
   - أضف Type Hints
   - اكتب Docstrings

3. **كتابة الاختبارات**
   - اختبارات وحدة للوظائف الجديدة
   - اختبارات تكامل للتفاعل بين المكونات

4. **تحديث التوثيق**
   - API documentation
   - README إذا لزم الأمر
   - CHANGELOG.md

5. **إنشاء Pull Request**

### مثال: إضافة نموذج جديد
```python
# src/radiya/models/new_model.py
from typing import Dict, Any
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

class ExtraTreesModel:
    """نموذج Extra Trees للتنبؤ بالانسحاب"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        تدريب نموذج Extra Trees
        
        Args:
            X: الميزات
            y: التسميات
            
        Returns:
            معلومات عن التدريب
        """
        self.model = ExtraTreesClassifier(
            n_estimators=100,
            random_state=self.random_state
        )
        
        self.model.fit(X, y)
        
        return {
            'model_type': 'ExtraTrees',
            'n_features': X.shape[1],
            'n_estimators': 100
        }
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """التنبؤ"""
        if self.model is None:
            raise ValueError("النموذج غير مدرب")
            
        return pd.Series(self.model.predict(X))
```

```python
# tests/unit/test_new_model.py
import pytest
import pandas as pd
from src.radiya.models.new_model import ExtraTreesModel

class TestExtraTreesModel:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4]
        }), pd.Series([0, 1, 0, 1])
    
    def test_model_training(self, sample_data):
        X, y = sample_data
        model = ExtraTreesModel()
        
        result = model.train(X, y)
        
        assert result['model_type'] == 'ExtraTrees'
        assert model.model is not None
```

---

## 6. إضافة API endpoint جديد

### إنشاء مسار جديد
```python
# src/radiya/api/routes/new_feature.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter(prefix="/new-feature", tags=["new-feature"])

class NewFeatureRequest(BaseModel):
    """نموذج طلب الميزة الجديدة"""
    data: List[Dict[str, Any]]
    parameters: Dict[str, Any] = {}

class NewFeatureResponse(BaseModel):
    """نموذج استجابة الميزة الجديدة"""
    success: bool
    results: List[Dict[str, Any]]
    message: str = ""

@router.post("/process", response_model=NewFeatureResponse)
async def process_new_feature(request: NewFeatureRequest):
    """معالجة الميزة الجديدة"""
    try:
        # منطق المعالجة
        results = []
        for item in request.data:
            # معالجة كل عنصر
            processed = {"processed": True, "item": item}
            results.append(processed)
            
        return NewFeatureResponse(
            success=True,
            results=results,
            message="تمت المعالجة بنجاح"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في المعالجة: {str(e)}"
        )
```

```python
# تحديث src/radiya/api/main.py
from .routes import prediction, analysis, upload, scheduler, new_feature

app.include_router(new_feature.router, prefix="/api/v1")
```

### اختبار API endpoint
```python
# tests/integration/test_new_feature_api.py
import pytest
from fastapi.testclient import TestClient
from src.radiya.api.main import app

client = TestClient(app)

def test_process_new_feature():
    """اختبار معالجة الميزة الجديدة"""
    request_data = {
        "data": [
            {"id": 1, "value": "test1"},
            {"id": 2, "value": "test2"}
        ],
        "parameters": {"option": "default"}
    }
    
    response = client.post("/api/v1/new-feature/process", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["results"]) == 2
```

---

## 7. إدارة قاعدة البيانات

### نماذج البيانات
```python
# src/radiya/database/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class User(Base):
    """نموذج المستخدم"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    gender = Column(String)
    level = Column(String)  # free or paid
    registration_date = Column(DateTime, default=func.now())
    last_activity = Column(DateTime)
    is_churned = Column(Boolean, default=False)
    churn_date = Column(DateTime, nullable=True)
    
class Session(Base):
    """نموذج الجلسة"""
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True)
    songs_played = Column(Integer, default=0)
    duration_minutes = Column(Float, default=0.0)
```

### Migration Scripts
```python
# migrations/create_tables.py
from src.radiya.database.models import Base
from sqlalchemy import create_engine

def create_tables(database_url: str):
    """إنشاء جداول قاعدة البيانات"""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    print("تم إنشاء الجداول بنجاح")

if __name__ == "__main__":
    create_tables("sqlite:///./radiya.db")
```

---

## 8. التصحيح والتطوير

### إعداد Logging
```python
# src/radiya/utils/logger.py
import logging
import sys
from pathlib import Path

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """إعداد نظام السجلات"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
```

### تصحيح الأخطاء
```python
# في الكود
import logging
logger = logging.getLogger(__name__)

def problematic_function(data):
    logger.debug(f"معالجة البيانات: {len(data)} عنصر")
    
    try:
        result = process_data(data)
        logger.info("تمت المعالجة بنجاح")
        return result
    except Exception as e:
        logger.error(f"خطأ في المعالجة: {e}", exc_info=True)
        raise
```

### استخدام Python Debugger
```python
# إضافة breakpoint
import pdb; pdb.set_trace()

# أو استخدام breakpoint() في Python 3.7+
breakpoint()
```

---

## 9. الأداء والتحسين

### Profiling
```python
# استخدام cProfile
import cProfile
import pstats

def profile_function():
    """قياس أداء دالة"""
    pr = cProfile.Profile()
    pr.enable()
    
    # الكود المراد قياس أداؤه
    result = expensive_operation()
    
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative').print_stats(10)
    
    return result
```

### Memory Optimization
```python
# استخدام Memory Profiler
from memory_profiler import profile

@profile
def memory_intensive_function():
    """دالة تستهلك ذاكرة كثيرة"""
    large_list = [i for i in range(1000000)]
    # معالجة البيانات
    return processed_data
```

### Caching
```python
# استخدام functools.lru_cache
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(parameter):
    """حساب مكلف مع تخزين مؤقت"""
    # حسابات معقدة
    return result
```

---

## 10. Git Workflow

### Branch Strategy
```
main                 # الفرع الرئيسي
├── develop          # فرع التطوير
├── feature/xyz      # فروع الميزات
├── hotfix/abc       # فروع الإصلاحات العاجلة
└── release/v1.x     # فروع الإصدارات
```

### Commit Messages
```
feat: إضافة نموذج Extra Trees للتنبؤ
fix: إصلاح خطأ في حساب معدل الانسحاب  
docs: تحديث توثيق API للميزة الجديدة
test: إضافة اختبارات للمجدول التلقائي
refactor: إعادة هيكلة كود هندسة الميزات
style: تحسين تنسيق الكود باستخدام black
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.1.6
  hooks:
    - id: ruff
      args: [--fix, --exit-non-zero-on-fix]
    - id: ruff-format

- repo: https://github.com/psf/black
  rev: 23.11.0
  hooks:
    - id: black

- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v4.5.0
  hooks:
    - id: trailing-whitespace
    - id: end-of-file-fixer
    - id: check-yaml
```

---

## 11. المساهمة في المشروع

### إرشادات المساهمة
1. **Fork المشروع** على GitHub
2. **إنشاء فرع جديد** للميزة/الإصلاح
3. **كتابة كود عالي الجودة** مع اختبارات
4. **تشغيل جميع الاختبارات** قبل الإرسال
5. **إنشاء Pull Request** مع وصف مفصل

### Code Review Checklist
- [ ] الكود يتبع معايير المشروع
- [ ] جميع الاختبارات تمر بنجاح  
- [ ] تم إضافة اختبارات للكود الجديد
- [ ] التوثيق محدث
- [ ] لا توجد أخطاء أمنية
- [ ] الأداء مقبول

---

## 12. الموارد والمراجع

### التوثيق التقني
- **FastAPI**: https://fastapi.tiangolo.com/
- **SQLAlchemy**: https://docs.sqlalchemy.org/
- **Pandas**: https://pandas.pydata.org/docs/
- **Scikit-learn**: https://scikit-learn.org/stable/

### أدوات التطوير
- **pytest**: https://docs.pytest.org/
- **ruff**: https://docs.astral.sh/ruff/
- **black**: https://black.readthedocs.io/
- **mypy**: https://mypy.readthedocs.io/

### الاتصال والدعم
- **GitHub Issues**: للإبلاغ عن مشاكل
- **Discussions**: للنقاشات العامة  
- **Email**: team@radiya.ai
- **Documentation**: https://radiya-docs.ai

---

هذا الدليل يوفر الأساسيات اللازمة لبدء التطوير والمساهمة في مشروع رضية. نرحب بجميع المساهمات والاقتراحات لتحسين المشروع!