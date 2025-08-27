#!/usr/bin/env python3
"""
سكريبت اختبار API لمشروع رضية
"""

import requests
import json
import time
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"

def test_api_endpoints():
    """اختبار جميع نقاط النهاية"""
    
    print("🧪 اختبار API لمشروع رضية")
    print("=" * 50)
    
    # اختبار الصفحة الرئيسية
    print("1. اختبار الصفحة الرئيسية...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        assert response.status_code == 200
        print("✅ الصفحة الرئيسية تعمل")
    except Exception as e:
        print(f"❌ خطأ في الصفحة الرئيسية: {e}")
    
    # اختبار فحص الصحة
    print("\n2. اختبار فحص الصحة...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        print("✅ فحص الصحة يعمل")
        print(f"   الحالة: {response.json()}")
    except Exception as e:
        print(f"❌ خطأ في فحص الصحة: {e}")
    
    # اختبار حالة النموذج
    print("\n3. اختبار حالة النموذج...")
    try:
        response = requests.get(f"{API_BASE_URL}/model/status")
        assert response.status_code == 200
        status = response.json()
        print("✅ حالة النموذج تعمل")
        print(f"   النموذج محمل: {status['model_loaded']}")
    except Exception as e:
        print(f"❌ خطأ في حالة النموذج: {e}")
    
    # اختبار التنبؤ (إذا كان النموذج محمل)
    print("\n4. اختبار التنبؤ...")
    try:
        # بيانات تجريبية
        test_data = {
            "userId": "test_user_123",
            "total_events": 150,
            "unique_sessions": 20,
            "active_days": 15,
            "songs_played": 100,
            "unique_songs": 80,
            "unique_artists": 50,
            "total_listening_time": 5000.0,
            "thumbs_up": 10,
            "thumbs_down": 2,
            "add_to_playlist": 5,
            "final_level_paid": 1,
            "events_per_session": 7.5,
            "songs_per_session": 5.0,
            "interaction_rate": 0.17
        }
        
        response = requests.post(f"{API_BASE_URL}/predict", json=test_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ التنبؤ يعمل")
            print(f"   المستخدم: {result['userId']}")
            print(f"   احتمالية الانسحاب: {result['churn_probability']:.3f}")
            print(f"   مستوى المخاطرة: {result['risk_level']}")
        else:
            print(f"⚠️ التنبؤ لا يعمل: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ خطأ في التنبؤ: {e}")
    
    # اختبار التنبؤ المجمع
    print("\n5. اختبار التنبؤ المجمع...")
    try:
        batch_data = {
            "users": [test_data, {**test_data, "userId": "test_user_456"}]
        }
        
        response = requests.post(f"{API_BASE_URL}/predict/batch", json=batch_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ التنبؤ المجمع يعمل")
            print(f"   المستخدمين: {result['total_users']}")
            print(f"   عالي المخاطرة: {result['high_risk_count']}")
            print(f"   وقت المعالجة: {result['processing_time']:.3f}s")
        else:
            print(f"⚠️ التنبؤ المجمع لا يعمل: {response.status_code}")
    except Exception as e:
        print(f"❌ خطأ في التنبؤ المجمع: {e}")
    
    print("\n" + "=" * 50)
    print("انتهى اختبار API")

def performance_test():
    """اختبار الأداء"""
    print("\n⚡ اختبار الأداء")
    print("-" * 30)
    
    test_data = {
        "userId": "perf_test_user",
        "total_events": 150,
        "unique_sessions": 20,
        "active_days": 15,
        "songs_played": 100,
        "unique_songs": 80,
        "unique_artists": 50,
        "total_listening_time": 5000.0,
        "thumbs_up": 10,
        "thumbs_down": 2,
        "add_to_playlist": 5,
        "final_level_paid": 1,
        "events_per_session": 7.5,
        "songs_per_session": 5.0,
        "interaction_rate": 0.17
    }
    
    # اختبار سرعة التنبؤ
    times = []
    for i in range(10):
        start_time = time.time()
        try:
            response = requests.post(f"{API_BASE_URL}/predict", json=test_data)
            if response.status_code == 200:
                end_time = time.time()
                times.append(end_time - start_time)
        except:
            pass
    
    if times:
        print(f"متوسط زمن الاستجابة: {sum(times)/len(times):.3f}s")
        print(f"أسرع استجابة: {min(times):.3f}s")
        print(f"أبطأ استجابة: {max(times):.3f}s")
    else:
        print("❌ فشل اختبار الأداء")

if __name__ == "__main__":
    test_api_endpoints()
    performance_test()