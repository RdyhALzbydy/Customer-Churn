# دليل النشر - مشروع رضية

## نظرة عامة
دليل شامل لنشر مشروع رضية في بيئات التطوير والإنتاج باستخدام Docker وأدوات الأتمتة الحديثة.

---

## 1. متطلبات النظام

### الحد الأدنى للمتطلبات
- **المعالج**: 2 نواة CPU
- **الذاكرة**: 4GB RAM
- **التخزين**: 20GB مساحة فارغة
- **الشبكة**: اتصال إنترنت مستقر

### المتطلبات الموصى بها (الإنتاج)
- **المعالج**: 4+ نواة CPU
- **الذاكرة**: 8GB+ RAM
- **التخزين**: 50GB+ SSD
- **الشبكة**: خطوط متعددة للتوافر العالي

### البرامج المطلوبة
- **Docker**: 20.0+
- **Docker Compose**: 2.0+
- **Python**: 3.9+ (للتطوير)
- **Git**: أحدث إصدار
- **uv**: مدير الحزم (اختياري)

---

## 2. إعداد البيئة

### تحميل المشروع
```bash
# استنساخ المشروع
git clone https://github.com/your-repo/radiya-project.git
cd radiya-project

# التحقق من الفروع
git branch -a
git checkout main
```

### إعداد متغيرات البيئة
```bash
# إنشاء ملف البيئة
cp .env.example .env

# تحرير المتغيرات
nano .env
```

**محتوى `.env`:**
```bash
# إعدادات عامة
PROJECT_NAME=radiya
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# إعدادات قاعدة البيانات
DATABASE_URL=sqlite:///./radiya.db
MLFLOW_TRACKING_URI=sqlite:///./mlflow.db

# إعدادات الخادم
HOST=0.0.0.0
PORT=8000
WORKERS=4

# إعدادات الأمان
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com

# إعدادات MLflow
MLFLOW_BACKEND_STORE_URI=sqlite:///./mlflow.db
MLFLOW_ARTIFACT_ROOT=./mlruns

# إعدادات إعادة التدريب
RETRAINING_ENABLED=true
RETRAINING_FREQUENCY=weekly
PERFORMANCE_THRESHOLD=0.8
```

---

## 3. النشر باستخدام Docker

### البناء والتشغيل السريع
```bash
# بناء وتشغيل جميع الخدمات
docker-compose up --build -d

# التحقق من حالة الخدمات
docker-compose ps

# عرض السجلات
docker-compose logs -f radiya-api
```

### الخدمات المتاحة
- **radiya-api**: `http://localhost:8000`
- **mlflow-ui**: `http://localhost:5000`
- **nginx** (في الإنتاج): `http://localhost:80`

### أوامر إدارية مفيدة
```bash
# إيقاف الخدمات
docker-compose down

# إعادة البناء الكامل
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d

# تنفيذ أوامر داخل الحاوية
docker-compose exec radiya-api bash

# نسخ ملفات من/إلى الحاوية
docker-compose cp ./local-file radiya-api:/app/
docker-compose cp radiya-api:/app/reports ./reports

# تنظيف الموارد غير المستخدمة
docker system prune -f
```

---

## 4. النشر في البيئات المختلفة

### أ. بيئة التطوير (Development)
```bash
# استخدام إعدادات التطوير
docker-compose -f docker-compose.dev.yml up --build -d

# أو التشغيل المحلي
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# أو .venv\Scripts\activate  # Windows

pip install -e ".[dev]"
uvicorn src.radiya.api.main:app --reload --host 0.0.0.0 --port 8000
```

### ب. بيئة الاختبار (Staging)
```bash
# إعداد بيئة الاختبار
cp .env.staging .env
docker-compose -f docker-compose.staging.yml up -d

# تشغيل الاختبارات
docker-compose exec radiya-api pytest tests/
```

### ج. بيئة الإنتاج (Production)
```bash
# إعداد الإنتاج
cp .env.production .env

# التحقق من الإعدادات
docker-compose -f docker-compose.prod.yml config

# النشر
docker-compose -f docker-compose.prod.yml up -d

# مراقبة الحالة
docker-compose -f docker-compose.prod.yml logs -f
```

---

## 5. إعدادات قاعدة البيانات

### SQLite (التطوير)
```bash
# لا يحتاج إعداد خاص، سيتم الإنشاء تلقائياً
```

### PostgreSQL (الإنتاج)
```yaml
# في docker-compose.prod.yml
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: radiya
      POSTGRES_USER: radiya_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

**تحديث `.env` للإنتاج:**
```bash
DATABASE_URL=postgresql://radiya_user:secure_password@postgres:5432/radiya
MLFLOW_BACKEND_STORE_URI=postgresql://radiya_user:secure_password@postgres:5432/mlflow
```

---

## 6. إعداد الويب سيرفر (Production)

### Nginx كـ Reverse Proxy
```nginx
# nginx/radiya.conf
server {
    listen 80;
    server_name yourdomain.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://radiya-api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /static {
        alias /app/static;
        expires 1y;
        add_header Cache-Control \"public, immutable\";
    }
}
```

### SSL/HTTPS إعداد
```bash
# باستخدام Let's Encrypt
certbot --nginx -d yourdomain.com

# أو إضافة SSL يدوياً في Nginx config
server {
    listen 443 ssl http2;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/private.key;
    
    # باقي الإعدادات...
}
```

---

## 7. المراقبة والصحة

### Health Checks
```bash
# التحقق من حالة API
curl http://localhost:8000/health

# التحقق من MLflow
curl http://localhost:5000/health

# مراقبة الموارد
docker stats
```

### Logging
```bash
# عرض سجلات مفصلة
docker-compose logs --tail=100 -f radiya-api

# حفظ السجلات في ملف
docker-compose logs radiya-api > radiya.log

# مراقبة أخطاء محددة
docker-compose logs radiya-api 2>&1 | grep ERROR
```

### Monitoring Script
```bash
#!/bin/bash
# monitoring.sh

echo "=== Radiya System Status ==="
echo "Date: $(date)"
echo ""

# Docker containers status
echo "Container Status:"
docker-compose ps

echo ""
echo "System Resources:"
docker stats --no-stream --format \"table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\"

echo ""
echo "API Health:"
curl -s http://localhost:8000/health | jq .

echo ""
echo "Disk Usage:"
df -h | grep -E '(Filesystem|/dev/)'
```

---

## 8. النسخ الاحتياطي والاستعادة

### النسخ الاحتياطي
```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./backups/$DATE"

mkdir -p $BACKUP_DIR

# نسخ قواعد البيانات
docker-compose exec -T postgres pg_dump -U radiya_user radiya > $BACKUP_DIR/database.sql

# نسخ النماذج والمصنوعات
cp -r ./models $BACKUP_DIR/
cp -r ./mlruns $BACKUP_DIR/
cp -r ./reports $BACKUP_DIR/

# ضغط النسخة الاحتياطية
tar -czf "backup_$DATE.tar.gz" -C ./backups $DATE

echo "Backup completed: backup_$DATE.tar.gz"
```

### الاستعادة
```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: ./restore.sh backup_YYYYMMDD_HHMMSS.tar.gz"
    exit 1
fi

# إيقاف الخدمات
docker-compose down

# فك الضغط
tar -xzf $BACKUP_FILE

# استعادة قاعدة البيانات
docker-compose up -d postgres
sleep 10
docker-compose exec -T postgres psql -U radiya_user -d radiya < ./database.sql

# استعادة الملفات
cp -r ./models/* ./models/
cp -r ./mlruns/* ./mlruns/

# إعادة تشغيل الخدمات
docker-compose up -d

echo "Restore completed from $BACKUP_FILE"
```

---

## 9. التحديث والتطوير

### تحديث النظام
```bash
# سحب أحدث التحديثات
git pull origin main

# إعادة بناء الصورة
docker-compose build radiya-api

# تحديث تدريجي (zero-downtime)
docker-compose up -d --scale radiya-api=2
sleep 30
docker-compose up -d --scale radiya-api=1
```

### Hot Reload للتطوير
```bash
# ربط الكود المحلي مع الحاوية
docker-compose -f docker-compose.dev.yml up -d

# أو تشغيل محلي مع مراقبة التغييرات
uvicorn src.radiya.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 10. حل المشاكل الشائعة

### مشاكل الذاكرة
```bash
# زيادة memory limit
docker-compose run --memory=2g radiya-api

# أو في docker-compose.yml
services:
  radiya-api:
    deploy:
      resources:
        limits:
          memory: 2G
```

### مشاكل الأذونات
```bash
# إصلاح أذونات الملفات
sudo chown -R $USER:$USER ./models ./reports ./mlruns

# داخل الحاوية
docker-compose exec radiya-api chown -R app:app /app
```

### مشاكل الشبكة
```bash
# إعادة إنشاء الشبكة
docker-compose down
docker network prune
docker-compose up -d
```

### مشاكل قاعدة البيانات
```bash
# إعادة تعيين قاعدة البيانات
docker-compose down -v
docker volume prune
docker-compose up -d
```

---

## 11. الأمان

### إعدادات الأمان الأساسية
```bash
# تغيير كلمات المرور الافتراضية
sed -i 's/default_password/$(openssl rand -base64 32)/' .env

# تحديد الشبكات المسموحة
iptables -A INPUT -p tcp --dport 8000 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -j DROP
```

### SSL/TLS
```bash
# إعداد HTTPS
docker run --rm -it \
  -v $(pwd)/certs:/etc/letsencrypt \
  certbot/certbot certonly \
  --webroot -w /var/www/html \
  -d yourdomain.com
```

---

## 12. التوسع (Scaling)

### التوسع الأفقي
```bash
# زيادة عدد المثيلات
docker-compose up -d --scale radiya-api=3

# مع Load Balancer
# في docker-compose.yml
services:
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
    depends_on:
      - radiya-api
```

### مراقبة الأداء
```bash
# استخدام htop لمراقبة الموارد
htop

# أو باستخدام Docker stats
watch docker stats
```

---

## 13. الخلاصة والموارد

### قائمة فحص النشر
- [ ] تم تكوين متغيرات البيئة
- [ ] تم اختبار الاتصال بقاعدة البيانات  
- [ ] تم تشغيل الخدمات بنجاح
- [ ] تم التحقق من Health Checks
- [ ] تم إعداد النسخ الاحتياطية
- [ ] تم تكوين المراقبة
- [ ] تم اختبار العمليات الأساسية

### موارد مفيدة
- **Docker Documentation**: https://docs.docker.com/
- **FastAPI Deployment**: https://fastapi.tiangolo.com/deployment/
- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **Nginx Configuration**: https://nginx.org/en/docs/

### الدعم والتواصل
- **Issues**: https://github.com/your-repo/issues
- **Discussions**: https://github.com/your-repo/discussions
- **Email**: team@radiya.ai