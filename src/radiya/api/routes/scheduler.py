"""
واجهة برمجة التطبيقات لإدارة مجدول إعادة التدريب
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime

from radiya.scheduler import (
    start_automated_retraining,
    stop_automated_retraining, 
    get_retraining_status
)

router = APIRouter(prefix="/scheduler", tags=["Automated Retraining"])

class SchedulerConfig(BaseModel):
    """إعدادات المجدول"""
    retraining_frequency: str = "weekly"  # daily, weekly, monthly
    min_new_data_threshold: int = 1000
    performance_threshold: float = 0.8
    data_drift_threshold: float = 0.1
    backup_models: bool = True
    notification_enabled: bool = True

class SchedulerResponse(BaseModel):
    """استجابة حالة المجدول"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

@router.post("/start", response_model=SchedulerResponse)
async def start_scheduler(config: Optional[SchedulerConfig] = None):
    """تشغيل مجدول إعادة التدريب التلقائي"""
    try:
        config_dict = config.model_dump() if config else None
        start_automated_retraining(config_dict)
        
        return SchedulerResponse(
            success=True,
            message="تم تشغيل مجدول إعادة التدريب التلقائي بنجاح"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في تشغيل المجدول: {str(e)}"
        )

@router.post("/stop", response_model=SchedulerResponse)
async def stop_scheduler():
    """إيقاف مجدول إعادة التدريب التلقائي"""
    try:
        stop_automated_retraining()
        
        return SchedulerResponse(
            success=True,
            message="تم إيقاف مجدول إعادة التدريب التلقائي"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في إيقاف المجدول: {str(e)}"
        )

@router.get("/status", response_model=SchedulerResponse)
async def get_scheduler_status():
    """الحصول على حالة مجدول إعادة التدريب"""
    try:
        status = get_retraining_status()
        
        return SchedulerResponse(
            success=True,
            message="تم جلب حالة المجدول بنجاح",
            data=status
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في جلب حالة المجدول: {str(e)}"
        )

@router.put("/config", response_model=SchedulerResponse)
async def update_scheduler_config(config: SchedulerConfig):
    """تحديث إعدادات مجدول إعادة التدريب"""
    try:
        # إيقاف المجدول الحالي
        stop_automated_retraining()
        
        # إعادة تشغيل بالإعدادات الجديدة
        start_automated_retraining(config.model_dump())
        
        return SchedulerResponse(
            success=True,
            message="تم تحديث إعدادات المجدول بنجاح"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في تحديث إعدادات المجدول: {str(e)}"
        )

@router.post("/trigger", response_model=SchedulerResponse)
async def trigger_retraining():
    """تشغيل إعادة التدريب يدوياً"""
    try:
        from radiya.scheduler import retraining_scheduler
        
        # تشغيل فحص إعادة التدريب مباشرة
        retraining_scheduler._check_and_retrain()
        
        return SchedulerResponse(
            success=True,
            message="تم تشغيل عملية إعادة التدريب يدوياً"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في تشغيل إعادة التدريب: {str(e)}"
        )