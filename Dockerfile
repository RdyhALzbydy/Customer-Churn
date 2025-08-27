FROM python:3.9-slim

# تثبيت متطلبات النظام
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# إنشاء مستخدم غير root
RUN useradd --create-home --shell /bin/bash radiya

# تعيين مجلد العمل
WORKDIR /app

# نسخ متطلبات Python
COPY requirements.txt .

# تثبيت المتطلبات
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# نسخ ملفات المشروع
COPY . .

# تغيير المالك للملفات
RUN chown -R radiya:radiya /app

# التبديل للمستخدم العادي
USER radiya

# إنشاء المجلدات المطلوبة
RUN mkdir -p data/raw data/processed models/saved_models models/scalers logs reports/figures reports/metrics

# تعيين متغيرات البيئة
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# كشف المنفذ
EXPOSE 8000

# فحص الصحة
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# الأمر الافتراضي
CMD ["uvicorn", "src.radiya.api.main:app", "--host", "0.0.0.0", "--port", "8000"]