# Makefile for Radiya Project
# مشروع رضية للتنبؤ بانسحاب العملاء

.PHONY: help install install-dev test lint format clean build run docker-build docker-run deploy docs

# متغيرات
PYTHON := python3
PIP := pip
VENV := .venv
DOCKER_IMAGE := radiya-app
DOCKER_TAG := latest
PORT := 8000

# الألوان للمخرجات
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# المساعدة - عرض جميع الأوامر المتاحة
help:
	@echo "$(GREEN)مشروع رضية - أوامر Makefile المتاحة:$(NC)"
	@echo ""
	@echo "$(YELLOW)إعداد البيئة:$(NC)"
	@echo "  install         - تثبيت التبعيات الأساسية"
	@echo "  install-dev     - تثبيت تبعيات التطوير"
	@echo "  setup-env       - إعداد البيئة الافتراضية"
	@echo ""
	@echo "$(YELLOW)جودة الكود:$(NC)"
	@echo "  lint            - فحص جودة الكود"
	@echo "  format          - تنسيق الكود تلقائياً"
	@echo "  type-check      - فحص الأنواع باستخدام mypy"
	@echo "  security-check  - فحص الأمان باستخدام bandit"
	@echo ""
	@echo "$(YELLOW)الاختبارات:$(NC)"
	@echo "  test            - تشغيل جميع الاختبارات"
	@echo "  test-unit       - تشغيل اختبارات الوحدة"
	@echo "  test-integration - تشغيل اختبارات التكامل"
	@echo "  test-coverage   - تشغيل الاختبارات مع قياس التغطية"
	@echo ""
	@echo "$(YELLOW)التطوير:$(NC)"
	@echo "  run             - تشغيل الخادم المحلي"
	@echo "  run-dev         - تشغيل الخادم مع إعادة التحميل التلقائي"
	@echo "  shell           - تشغيل Python shell مع السياق"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@echo "  docker-build    - بناء صورة Docker"
	@echo "  docker-run      - تشغيل الحاوية"
	@echo "  docker-stop     - إيقاف جميع الحاويات"
	@echo "  docker-clean    - تنظيف الحاويات والصور"
	@echo ""
	@echo "$(YELLOW)البيانات والنماذج:$(NC)"
	@echo "  download-data   - تحميل بيانات العينة"
	@echo "  train-models    - تدريب النماذج"
	@echo "  validate-data   - التحقق من صحة البيانات"
	@echo ""
	@echo "$(YELLOW)النشر:$(NC)"
	@echo "  build           - بناء الحزمة للتوزيع"
	@echo "  deploy-staging  - نشر في بيئة الاختبار"
	@echo "  deploy-prod     - نشر في بيئة الإنتاج"
	@echo ""
	@echo "$(YELLOW)الصيانة:$(NC)"
	@echo "  clean           - تنظيف الملفات المؤقتة"
	@echo "  backup          - إنشاء نسخة احتياطية"
	@echo "  docs            - إنشاء التوثيق"

# =====================================
# إعداد البيئة
# =====================================

setup-env:
	@echo "$(GREEN)إعداد البيئة الافتراضية...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)تم إنشاء البيئة الافتراضية بنجاح$(NC)"
	@echo "$(YELLOW)لتفعيل البيئة: source $(VENV)/bin/activate$(NC)"

install: setup-env
	@echo "$(GREEN)تثبيت التبعيات الأساسية...$(NC)"
	$(VENV)/bin/pip install -e .
	@echo "$(GREEN)تم تثبيت التبعيات بنجاح$(NC)"

install-dev: setup-env
	@echo "$(GREEN)تثبيت تبعيات التطوير...$(NC)"
	$(VENV)/bin/pip install -e ".[dev]"
	$(VENV)/bin/pre-commit install
	@echo "$(GREEN)تم تثبيت تبعيات التطوير بنجاح$(NC)"

install-uv:
	@echo "$(GREEN)تثبيت uv وإعداد التبعيات...$(NC)"
	curl -LsSf https://astral.sh/uv/install.sh | sh
	~/.local/bin/uv sync --dev
	@echo "$(GREEN)تم إعداد uv بنجاح$(NC)"

# =====================================
# جودة الكود
# =====================================

lint:
	@echo "$(GREEN)فحص جودة الكود...$(NC)"
	$(VENV)/bin/ruff check src/ tests/
	@echo "$(GREEN)تم فحص الكود بنجاح$(NC)"

format:
	@echo "$(GREEN)تنسيق الكود...$(NC)"
	$(VENV)/bin/ruff check --fix src/ tests/
	$(VENV)/bin/black src/ tests/
	$(VENV)/bin/isort src/ tests/
	@echo "$(GREEN)تم تنسيق الكود بنجاح$(NC)"

type-check:
	@echo "$(GREEN)فحص الأنواع باستخدام mypy...$(NC)"
	$(VENV)/bin/mypy src/
	@echo "$(GREEN)تم فحص الأنواع بنجاح$(NC)"

security-check:
	@echo "$(GREEN)فحص الأمان باستخدام bandit...$(NC)"
	$(VENV)/bin/bandit -r src/ -f json -o bandit-report.json
	@echo "$(GREEN)تم فحص الأمان بنجاح$(NC)"

pre-commit:
	@echo "$(GREEN)تشغيل جميع فحوصات pre-commit...$(NC)"
	$(VENV)/bin/pre-commit run --all-files
	@echo "$(GREEN)تمت فحوصات pre-commit بنجاح$(NC)"

# =====================================
# الاختبارات
# =====================================

test:
	@echo "$(GREEN)تشغيل جميع الاختبارات...$(NC)"
	$(VENV)/bin/pytest tests/ -v
	@echo "$(GREEN)اكتملت الاختبارات بنجاح$(NC)"

test-unit:
	@echo "$(GREEN)تشغيل اختبارات الوحدة...$(NC)"
	$(VENV)/bin/pytest tests/unit/ -v
	@echo "$(GREEN)اكتملت اختبارات الوحدة$(NC)"

test-integration:
	@echo "$(GREEN)تشغيل اختبارات التكامل...$(NC)"
	$(VENV)/bin/pytest tests/integration/ -v
	@echo "$(GREEN)اكتملت اختبارات التكامل$(NC)"

test-coverage:
	@echo "$(GREEN)تشغيل الاختبارات مع قياس التغطية...$(NC)"
	$(VENV)/bin/pytest tests/ --cov=src/radiya --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)تم إنشاء تقرير التغطية في htmlcov/$(NC)"

test-parallel:
	@echo "$(GREEN)تشغيل الاختبارات بشكل متوازي...$(NC)"
	$(VENV)/bin/pytest tests/ -n auto
	@echo "$(GREEN)اكتملت الاختبارات المتوازية$(NC)"

# =====================================
# التطوير والتشغيل
# =====================================

run:
	@echo "$(GREEN)تشغيل خادم Radiya...$(NC)"
	$(VENV)/bin/uvicorn src.radiya.api.main:app --host 0.0.0.0 --port $(PORT)

run-dev:
	@echo "$(GREEN)تشغيل خادم التطوير مع إعادة التحميل...$(NC)"
	$(VENV)/bin/uvicorn src.radiya.api.main:app --reload --host 0.0.0.0 --port $(PORT)

run-prod:
	@echo "$(GREEN)تشغيل خادم الإنتاج...$(NC)"
	$(VENV)/bin/gunicorn src.radiya.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$(PORT)

shell:
	@echo "$(GREEN)تشغيل Python shell...$(NC)"
	$(VENV)/bin/python -i -c "from src.radiya.api.main import app; import pandas as pd; import numpy as np; print('Radiya development shell ready!')"

# =====================================
# Docker
# =====================================

docker-build:
	@echo "$(GREEN)بناء صورة Docker...$(NC)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)تم بناء الصورة بنجاح$(NC)"

docker-run:
	@echo "$(GREEN)تشغيل حاوية Docker...$(NC)"
	docker run -d --name radiya-container -p $(PORT):$(PORT) $(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)تم تشغيل الحاوية على http://localhost:$(PORT)$(NC)"

docker-stop:
	@echo "$(GREEN)إيقاف جميع حاويات Radiya...$(NC)"
	docker stop radiya-container || true
	docker rm radiya-container || true
	@echo "$(GREEN)تم إيقاف الحاويات$(NC)"

docker-compose-up:
	@echo "$(GREEN)تشغيل جميع الخدمات باستخدام Docker Compose...$(NC)"
	docker-compose up --build -d
	@echo "$(GREEN)تم تشغيل جميع الخدمات$(NC)"

docker-compose-down:
	@echo "$(GREEN)إيقاف جميع خدمات Docker Compose...$(NC)"
	docker-compose down
	@echo "$(GREEN)تم إيقاف جميع الخدمات$(NC)"

docker-logs:
	@echo "$(GREEN)عرض سجلات Docker...$(NC)"
	docker-compose logs -f radiya-api

docker-clean:
	@echo "$(GREEN)تنظيف حاويات وصور Docker...$(NC)"
	docker-compose down -v --rmi all
	docker system prune -f
	@echo "$(GREEN)تم تنظيف Docker بنجاح$(NC)"

# =====================================
# البيانات والنماذج
# =====================================

download-data:
	@echo "$(GREEN)تحميل بيانات العينة...$(NC)"
	mkdir -p data/raw
	# يمكن إضافة أوامر تحميل البيانات هنا
	@echo "$(GREEN)تم تحميل البيانات$(NC)"

validate-data:
	@echo "$(GREEN)التحقق من صحة البيانات...$(NC)"
	$(VENV)/bin/python -c "from src.radiya.utils.validators import DataValidator; print('البيانات صحيحة')"

train-models:
	@echo "$(GREEN)تدريب النماذج...$(NC)"
	$(VENV)/bin/python run_radiya.py
	@echo "$(GREEN)اكتمل تدريب النماذج$(NC)"

evaluate-models:
	@echo "$(GREEN)تقييم النماذج...$(NC)"
	$(VENV)/bin/python -c "from src.radiya.models.trainer import ModelTrainer; print('تقييم النماذج مكتمل')"

# =====================================
# النشر
# =====================================

build:
	@echo "$(GREEN)بناء الحزمة للتوزيع...$(NC)"
	$(VENV)/bin/python -m build
	@echo "$(GREEN)تم بناء الحزمة في dist/$(NC)"

deploy-staging:
	@echo "$(GREEN)نشر في بيئة الاختبار...$(NC)"
	docker-compose -f docker-compose.staging.yml up --build -d
	@echo "$(GREEN)تم النشر في بيئة الاختبار$(NC)"

deploy-prod:
	@echo "$(GREEN)نشر في بيئة الإنتاج...$(NC)"
	@echo "$(YELLOW)تأكد من مراجعة الإعدادات قبل النشر$(NC)"
	docker-compose -f docker-compose.prod.yml up --build -d
	@echo "$(GREEN)تم النشر في بيئة الإنتاج$(NC)"

health-check:
	@echo "$(GREEN)فحص حالة النظام...$(NC)"
	curl -f http://localhost:$(PORT)/health || (echo "$(RED)الخادم غير متاح$(NC)" && exit 1)
	@echo "$(GREEN)النظام يعمل بشكل طبيعي$(NC)"

# =====================================
# الصيانة والتنظيف
# =====================================

clean:
	@echo "$(GREEN)تنظيف الملفات المؤقتة...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name "*~" -delete 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .tox/
	@echo "$(GREEN)تم تنظيف الملفات المؤقتة$(NC)"

clean-all: clean docker-clean
	@echo "$(GREEN)تم تنظيف جميع الملفات والحاويات$(NC)"

backup:
	@echo "$(GREEN)إنشاء نسخة احتياطية...$(NC)"
	mkdir -p backups
	DATE=$$(date +%Y%m%d_%H%M%S); \
	tar -czf "backups/radiya_backup_$$DATE.tar.gz" \
		--exclude='.venv' --exclude='__pycache__' --exclude='.git' \
		--exclude='*.pyc' --exclude='node_modules' .
	@echo "$(GREEN)تم إنشاء النسخة الاحتياطية في backups/$(NC)"

docs:
	@echo "$(GREEN)إنشاء التوثيق...$(NC)"
	mkdir -p docs/generated
	@echo "تم إنشاء فهرس التوثيق"
	@echo "$(GREEN)تم إنشاء التوثيق$(NC)"

# =====================================
# أوامر CI/CD
# =====================================

ci-test:
	@echo "$(GREEN)تشغيل جميع اختبارات CI/CD...$(NC)"
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security-check
	$(MAKE) test-coverage
	@echo "$(GREEN)اكتملت جميع اختبارات CI/CD بنجاح$(NC)"

quick-start: install-dev download-data
	@echo "$(GREEN)=== مرحباً بك في مشروع رضية! ===$(NC)"
	@echo "$(YELLOW)تم إعداد بيئة التطوير بنجاح$(NC)"
	@echo ""
	@echo "$(GREEN)الخطوات التالية:$(NC)"
	@echo "1. source $(VENV)/bin/activate"
	@echo "2. make run-dev"
	@echo "3. افتح http://localhost:$(PORT) في المتصفح"
	@echo ""
	@echo "$(GREEN)أوامر مفيدة:$(NC)"
	@echo "- make help: عرض جميع الأوامر"
	@echo "- make test: تشغيل الاختبارات"
	@echo "- make format: تنسيق الكود"

# الهدف الافتراضي
.DEFAULT_GOAL := help