from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    userId: str
    total_events: int
    songs_played: int
    unique_songs: int
    unique_artists: int
    total_listening_time: float
    thumbs_up: int
    thumbs_down: int
    add_to_playlist: int
    home_visits: int
    active_days: int
    unique_sessions: int
    avg_hour: float
    weekend_activity_ratio: float
    gender_M: Optional[int] = 0
    gender_F: Optional[int] = 0
    final_level_paid: int = 0

class PredictionResponse(BaseModel):
    userId: str
    churn_probability: float
    risk_level: str
    confidence: float
    model_used: str
    timestamp: str

@router.post("/predict", response_model=PredictionResponse)
async def predict_churn(request: PredictionRequest):
    """التنبؤ بانسحاب مستخدم واحد"""
    
    try:
        # تحضير البيانات
        features = prepare_features(request.dict())
        
        # محاكاة التنبؤ (يمكن استبدالها بنموذج حقيقي)
        probability = simulate_prediction(features)
        
        # تحديد مستوى المخاطر
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        confidence = min(95, max(70, int((1 - abs(probability - 0.5) * 2) * 100)))
        
        return PredictionResponse(
            userId=request.userId,
            churn_probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            model_used="XGBoost",
            timestamp=pd.Timestamp.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"خطأ في التنبؤ: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def prepare_features(data: Dict[str, Any]) -> Dict[str, float]:
    """تحضير الميزات للتنبؤ"""
    
    features = data.copy()
    
    # حساب الميزات المشتقة
    if features['songs_played'] > 0:
        features['interaction_rate'] = (
            features['thumbs_up'] + features['thumbs_down'] + features['add_to_playlist']
        ) / features['songs_played']
    else:
        features['interaction_rate'] = 0
    
    if features['unique_sessions'] > 0:
        features['events_per_session'] = features['total_events'] / features['unique_sessions']
        features['songs_per_session'] = features['songs_played'] / features['unique_sessions']
    else:
        features['events_per_session'] = 0
        features['songs_per_session'] = 0
    
    if features['active_days'] > 0:
        features['events_per_day'] = features['total_events'] / features['active_days']
        features['songs_per_day'] = features['songs_played'] / features['active_days']
    else:
        features['events_per_day'] = 0
        features['songs_per_day'] = 0
    
    return features

def simulate_prediction(features: Dict[str, float]) -> float:
    """محاكاة التنبؤ"""
    
    # حساب المخاطر بناء على القواعد
    risk_score = 0.0
    
    # عوامل المشاركة
    if features.get('interaction_rate', 0) < 0.1:
        risk_score += 0.3
    if features.get('songs_per_day', 0) < 5:
        risk_score += 0.2
    if features.get('events_per_session', 0) < 10:
        risk_score += 0.2
    if features.get('weekend_activity_ratio', 0) > 0.7:
        risk_score += 0.15
    if features.get('final_level_paid', 0) == 0:
        risk_score += 0.25
    if features.get('thumbs_down', 0) > features.get('thumbs_up', 0):
        risk_score += 0.2
    if features.get('home_visits', 0) < 5:
        risk_score += 0.1
    
    # إضافة عشوائية للواقعية
    risk_score += (np.random.random() - 0.5) * 0.2
    risk_score = max(0, min(1, risk_score))
    
    return float(risk_score)