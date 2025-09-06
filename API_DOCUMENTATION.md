# توثيق واجهة برمجة التطبيقات - مشروع رضية

## نظرة عامة
واجهة برمجة التطبيقات (API) لمشروع رضية توفر نقاط وصول شاملة للتنبؤ بانسحاب العملاء، إدارة البيانات، وأتمتة عمليات التدريب.

## العنوان الأساسي
```
http://localhost:8000/api/v1
```

## المصادقة
حالياً لا تتطلب المصادقة، ولكن يمكن إضافة JWT أو API Keys حسب الحاجة.

## تنسيق الاستجابة
جميع الاستجابات بتنسيق JSON مع البنية التالية:
```json
{
  "success": true,
  "message": "رسالة وصفية",
  "data": {},
  "timestamp": "2025-01-15T10:30:00Z"
}
```

---

## 1. مجموعة التنبؤ (Prediction Endpoints)

### 1.1 التنبؤ بانسحاب عميل واحد
**POST** `/prediction/predict`

التنبؤ باحتمالية انسحاب عميل واحد بناءً على الميزات المُرسلة.

**Request Body:**
```json
{
  "user_features": {
    "avg_songs_per_session": 15.5,
    "total_sessions": 120,
    "total_songs": 1860,
    "avg_session_duration": 45.2,
    "days_since_registration": 180,
    "level": "paid",
    "gender": "M",
    "subscription_days": 150,
    "thumbs_up_ratio": 0.75,
    "add_to_playlist_ratio": 0.25,
    "downgrade_events": 0,
    "error_rate": 0.02
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "تم التنبؤ بنجاح",
  "data": {
    "churn_probability": 0.23,
    "churn_prediction": "stable",
    "confidence_level": "high",
    "risk_factors": [
      {
        "factor": "low_engagement",
        "importance": 0.15,
        "description": "معدل تفاعل أقل من المتوسط"
      }
    ],
    "model_used": "RandomForest",
    "prediction_timestamp": "2025-01-15T10:30:00Z"
  }
}
```

### 1.2 التنبؤ الدفعي
**POST** `/prediction/batch_predict`

التنبؤ لعدة عملاء في طلب واحد.

**Request Body:**
```json
{
  "users": [
    {
      "user_id": "user_001",
      "features": { /* ميزات المستخدم */ }
    },
    {
      "user_id": "user_002", 
      "features": { /* ميزات المستخدم */ }
    }
  ]
}
```

### 1.3 تفسير التنبؤ
**POST** `/prediction/explain`

الحصول على تفسير مفصل لنتيجة التنبؤ.

**Response:**
```json
{
  "success": true,
  "data": {
    "feature_importance": {
      "avg_songs_per_session": 0.25,
      "subscription_days": 0.20,
      "thumbs_up_ratio": 0.18
    },
    "shap_values": {
      "base_value": 0.3,
      "feature_contributions": { /* مساهمة كل ميزة */ }
    }
  }
}
```

---

## 2. مجموعة تحميل البيانات (Upload Endpoints)

### 2.1 تحميل ملف بيانات
**POST** `/upload/file`

تحميل ملف بيانات للمعالجة والتدريب.

**Request:** Multipart Form Data
```
file: [JSON file]
```

**Response:**
```json
{
  "success": true,
  "message": "تم تحميل الملف بنجاح",
  "data": {
    "file_id": "upload_20250115_103000",
    "file_size": "125.5MB",
    "records_count": 225000,
    "validation_status": "passed",
    "processing_status": "completed"
  }
}
```

### 2.2 الحصول على تحليل البيانات
**GET** `/upload/analysis/{file_id}`

تحليل شامل للبيانات المحملة.

**Response:**
```json
{
  "success": true,
  "data": {
    "basic_stats": {
      "total_users": 22500,
      "total_sessions": 1125000,
      "date_range": {
        "start": "2024-01-01",
        "end": "2024-12-31"
      }
    },
    "churn_analysis": {
      "cancellation_rate": 23.1,
      "downgrade_rate": 21.8,
      "inactivity_rate": 49.8,
      "combined_rate": 40.9
    },
    "feature_distribution": {
      "gender": {"M": 55.2, "F": 44.8},
      "level": {"paid": 67.3, "free": 32.7}
    },
    "data_quality": {
      "missing_values": 0.02,
      "duplicate_records": 0.001,
      "anomalies_detected": 12
    }
  }
}
```

### 2.3 التحقق من صحة البيانات
**POST** `/upload/validate`

التحقق من صحة وجودة البيانات المحملة.

**Response:**
```json
{
  "success": true,
  "data": {
    "is_valid": true,
    "validation_results": {
      "schema_valid": true,
      "data_types_correct": true,
      "required_columns_present": true,
      "date_format_valid": true
    },
    "warnings": [
      "5 سجلات تحتوي على قيم مفقودة في العمود 'location'",
      "تم العثور على 3 قيم شاذة في 'session_duration'"
    ],
    "recommendations": [
      "استخدام SMOTE لموازنة البيانات",
      "تطبيق تطبيع للميزات الرقمية"
    ]
  }
}
```

---

## 3. مجموعة التحليل (Analysis Endpoints)

### 3.1 تحليل شامل للبيانات
**GET** `/analysis/comprehensive/{file_id}`

**Response:**
```json
{
  "success": true,
  "data": {
    "user_behavior": {
      "avg_session_duration": 42.5,
      "most_popular_pages": ["NextSong", "Home", "Thumbs Up"],
      "peak_usage_hours": [19, 20, 21]
    },
    "churn_patterns": {
      "high_risk_segments": [
        {
          "segment": "free_users_low_engagement",
          "size": 1500,
          "churn_rate": 65.4
        }
      ],
      "churn_triggers": [
        "billing_error_frequency",
        "session_gap_duration",
        "song_skip_rate"
      ]
    },
    "feature_insights": {
      "most_predictive": ["subscription_length", "song_diversity"],
      "least_predictive": ["browser_type", "device_model"]
    }
  }
}
```

### 3.2 تحليل الاتجاهات الزمنية
**GET** `/analysis/trends/{file_id}`

تحليل الاتجاهات الزمنية في سلوك المستخدمين.

### 3.3 تحليل التجزئة
**POST** `/analysis/segmentation`

تحليل تجزئة العملاء إلى مجموعات.

---

## 4. مجموعة المجدول (Scheduler Endpoints)

### 4.1 تشغيل المجدول التلقائي
**POST** `/scheduler/start`

**Request Body:**
```json
{
  "retraining_frequency": "weekly",
  "min_new_data_threshold": 1000,
  "performance_threshold": 0.8,
  "backup_models": true,
  "notification_enabled": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "تم تشغيل مجدول إعادة التدريب التلقائي بنجاح",
  "data": {
    "scheduler_id": "sched_20250115_103000",
    "next_run": "2025-01-22T02:00:00Z",
    "status": "running"
  }
}
```

### 4.2 إيقاف المجدول
**POST** `/scheduler/stop`

### 4.3 حالة المجدول
**GET** `/scheduler/status`

**Response:**
```json
{
  "success": true,
  "data": {
    "is_running": true,
    "last_training_time": "2025-01-15T02:00:00Z",
    "next_scheduled_run": "2025-01-22T02:00:00Z",
    "total_automated_trainings": 5,
    "last_training_results": {
      "models_trained": 16,
      "best_performance": 0.95,
      "training_duration": "45 minutes"
    }
  }
}
```

### 4.4 تشغيل إعادة التدريب يدوياً
**POST** `/scheduler/trigger`

تشغيل عملية إعادة التدريب فوراً.

### 4.5 تحديث إعدادات المجدول
**PUT** `/scheduler/config`

---

## 5. نقاط وصول المراقبة (Monitoring Endpoints)

### 5.1 حالة النظام
**GET** `/health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z",
  "services": {
    "api": "up",
    "mlflow": "up", 
    "database": "up",
    "scheduler": "up"
  },
  "performance": {
    "response_time_ms": 25,
    "memory_usage": "45%",
    "cpu_usage": "12%"
  }
}
```

### 5.2 معلومات النظام
**GET** `/info`

### 5.3 إحصائيات الاستخدام
**GET** `/metrics`

---

## أكواد الاستجابة (Response Codes)

| الكود | المعنى | الوصف |
|-------|--------|-------|
| 200 | OK | نجح الطلب |
| 201 | Created | تم إنشاء المورد |
| 400 | Bad Request | طلب غير صالح |
| 404 | Not Found | المورد غير موجود |
| 422 | Unprocessable Entity | بيانات غير صالحة |
| 500 | Internal Server Error | خطأ في الخادم |

---

## أمثلة الاستخدام

### Python
```python
import requests

# التنبؤ لعميل واحد
response = requests.post(
    'http://localhost:8000/api/v1/prediction/predict',
    json={
        'user_features': {
            'avg_songs_per_session': 15.5,
            'total_sessions': 120,
            # باقي الميزات...
        }
    }
)
result = response.json()
print(f"احتمالية الانسحاب: {result['data']['churn_probability']}")
```

### cURL
```bash
# تحميل ملف بيانات
curl -X POST "http://localhost:8000/api/v1/upload/file" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@customer_data.json"

# التنبؤ
curl -X POST "http://localhost:8000/api/v1/prediction/predict" \
  -H "Content-Type: application/json" \
  -d '{"user_features": {"avg_songs_per_session": 15.5}}'
```

### JavaScript
```javascript
// التنبؤ باستخدام Fetch API
fetch('http://localhost:8000/api/v1/prediction/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    user_features: {
      avg_songs_per_session: 15.5,
      total_sessions: 120,
      // باقي الميزات...
    }
  })
})
.then(response => response.json())
.then(data => {
  console.log('احتمالية الانسحاب:', data.data.churn_probability);
});
```

---

## حدود المعدل (Rate Limits)
- **التنبؤ الفردي**: 100 طلب/دقيقة
- **التنبؤ الدفعي**: 10 طلبات/دقيقة  
- **تحميل الملفات**: 5 تحميلات/ساعة
- **باقي النقاط**: 1000 طلب/ساعة

---

## الإصدارات
- **v1.0.0**: الإصدار الحالي
- **التوافق**: يتم الحفاظ على التوافق العكسي
- **التطوير**: إصدارات جديدة كل 3 أشهر

---

## الدعم والتواصل
للمساعدة أو الإبلاغ عن مشاكل:
- **البريد الإلكتروني**: team@radiya.ai
- **التوثيق التفاعلي**: http://localhost:8000/docs
- **GitHub Issues**: https://github.com/your-repo/issues