.PHONY: install setup data analysis train evaluate api test clean

# التثبيت والإعداد
install:
	pip install -r requirements.txt
	pip install -e .

setup:
	mkdir -p data/raw data/processed models/saved_models logs reports/figures
	
# تحليل البيانات
data:
	python scripts/run_analysis.py

analysis: data

# تدريب النماذج
train:
	python scripts/train_model.py

# تقييم النماذج
evaluate:
	python scripts/evaluate_models.py

# مقارنة شاملة
compare: train evaluate

# تشغيل API
api:
	uvicorn src.radiya.api.main:app --host 0.0.0.0 --port 8000 --reload

# تشغيل الاختبارات
test:
	pytest tests/ -v

# تنظيف الملفات المؤقتة
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache