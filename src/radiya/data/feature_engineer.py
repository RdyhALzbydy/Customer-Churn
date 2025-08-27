"""
هندسة الميزات لمشروع رضية - النسخة المصححة
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """كلاس هندسة الميزات لبيانات Sparkify"""
    
    def __init__(self):
        """تهيئة مهندس الميزات"""
        self.feature_names = []
        self.churn_definitions = {}
        
    def define_churn(self, df: pd.DataFrame, method: str = 'combined') -> pd.DataFrame:
        """تعريف المستخدمين المنسحبين بطرق مختلفة"""
        
        logger.info(f"تعريف الانسحاب بطريقة: {method}")
        
        all_users = df['userId'].unique()
        churn_labels = pd.DataFrame({
            'userId': all_users,
            'churned': 0
        })
        
        if method == 'cancellation':
            cancelled_users = df[df['page'] == 'Cancellation Confirmation']['userId'].unique()
            churn_labels.loc[churn_labels['userId'].isin(cancelled_users), 'churned'] = 1
            
        elif method == 'downgrade':
            downgrade_users = df[df['page'] == 'Submit Downgrade']['userId'].unique()
            churn_labels.loc[churn_labels['userId'].isin(downgrade_users), 'churned'] = 1
            
        elif method == 'combined':
            cancelled_users = df[df['page'] == 'Cancellation Confirmation']['userId'].unique()
            downgrade_users = df[df['page'] == 'Submit Downgrade']['userId'].unique()
            churn_users = set(cancelled_users) | set(downgrade_users)
            churn_labels.loc[churn_labels['userId'].isin(churn_users), 'churned'] = 1
            
        elif method == 'inactivity':
            last_activity = df.groupby('userId')['datetime'].max()
            max_date = df['datetime'].max()
            inactive_threshold = max_date - timedelta(days=7)
            inactive_users = last_activity[last_activity < inactive_threshold].index
            churn_labels.loc[churn_labels['userId'].isin(inactive_users), 'churned'] = 1
            
        else:
            raise ValueError(f"طريقة غير مدعومة: {method}")
        
        churned_count = churn_labels['churned'].sum()
        churn_rate = churned_count / len(churn_labels) * 100
        
        logger.info(f"المنسحبين: {churned_count:,}, المعدل: {churn_rate:.2f}%")
        
        self.churn_definitions[method] = churn_labels
        return churn_labels
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إنشاء جميع الميزات"""
        
        logger.info("بدء إنشاء الميزات الشاملة...")
        
        # إنشاء ميزات المستخدمين
        user_features = self.create_user_features(df)
        
        # ملء القيم المفقودة - للأرقام فقط
        numeric_columns = user_features.select_dtypes(include=[np.number]).columns
        user_features[numeric_columns] = user_features[numeric_columns].fillna(0)
        
        # استبدال القيم اللانهائية - للأرقام فقط
        user_features[numeric_columns] = user_features[numeric_columns].replace([np.inf, -np.inf], 0)
        
        # حفظ أسماء الميزات
        self.feature_names = [col for col in user_features.columns if col != 'userId']
        
        logger.info(f"تم إنشاء {len(self.feature_names)} ميزة نهائية")
        
        return user_features
    
    def create_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إنشاء ميزات المستخدمين"""
        
        logger.info("إنشاء ميزات المستخدمين...")
        
        user_features = []
        
        for user_id in df['userId'].unique():
            user_data = df[df['userId'] == user_id].sort_values('datetime')
            features = self._extract_user_features(user_data, user_id)
            user_features.append(features)
        
        features_df = pd.DataFrame(user_features)
        logger.info(f"تم إنشاء {len(features_df.columns)} ميزة لـ {len(features_df)} مستخدم")
        
        return features_df
    
    def _extract_user_features(self, user_data: pd.DataFrame, user_id: str) -> Dict:
        """استخراج ميزات مستخدم واحد"""
        
        features = {'userId': user_id}
        
        # الميزات الأساسية
        total_events = len(user_data)
        unique_sessions = user_data['sessionId'].nunique()
        active_days = user_data['date'].nunique()
        
        features.update({
            'total_events': total_events,
            'unique_sessions': unique_sessions,
            'active_days': active_days,
            'total_time_span_hours': max(1, (user_data['datetime'].max() - user_data['datetime'].min()).total_seconds() / 3600)
        })
        
        # ميزات الاستماع
        songs_data = user_data[user_data['page'] == 'NextSong']
        songs_count = len(songs_data)
        
        features.update({
            'songs_played': songs_count,
            'unique_songs': songs_data['song'].nunique() if songs_count > 0 else 0,
            'unique_artists': songs_data['artist'].nunique() if songs_count > 0 else 0,
            'total_listening_time': songs_data['length'].sum() if songs_count > 0 else 0,
            'avg_song_length': songs_data['length'].mean() if songs_count > 0 else 0,
            'std_song_length': songs_data['length'].std() if songs_count > 0 else 0
        })
        
        # ميزات التفاعل
        thumbs_up = len(user_data[user_data['page'] == 'Thumbs Up'])
        thumbs_down = len(user_data[user_data['page'] == 'Thumbs Down'])
        add_playlist = len(user_data[user_data['page'] == 'Add to Playlist'])
        
        features.update({
            'thumbs_up': thumbs_up,
            'thumbs_down': thumbs_down,
            'add_to_playlist': add_playlist,
            'add_friend': len(user_data[user_data['page'] == 'Add Friend']),
            'home_visits': len(user_data[user_data['page'] == 'Home']),
            'help_visits': len(user_data[user_data['page'] == 'Help']),
            'settings_visits': len(user_data[user_data['page'] == 'Settings']),
            'upgrade_visits': len(user_data[user_data['page'] == 'Upgrade']),
            'downgrade_visits': len(user_data[user_data['page'] == 'Submit Downgrade']),
            'logout_count': len(user_data[user_data['page'] == 'Logout']),
            'error_count': len(user_data[user_data['page'] == 'Error'])
        })
        
        # ميزات الاشتراك
        levels = user_data['level'].unique()
        features.update({
            'final_level_paid': 1 if user_data['level'].iloc[-1] == 'paid' else 0,
            'started_as_paid': 1 if user_data['level'].iloc[0] == 'paid' else 0,
            'level_changes': max(0, user_data['level'].nunique() - 1),
            'was_ever_paid': 1 if 'paid' in levels else 0,
            'subscription_tenure_days': max(1, (user_data['datetime'].max() - user_data['datetime'].min()).days)
        })
        
        # معلومات ديموغرافية
        gender = user_data['gender'].iloc[0] if pd.notna(user_data['gender'].iloc[0]) else 'Unknown'
        features.update({
            'gender_M': 1 if gender == 'M' else 0,
            'gender_F': 1 if gender == 'F' else 0,
        })
        
        # ميزات زمنية  
        features.update({
            'avg_hour': user_data['hour'].mean(),
            'std_hour': user_data['hour'].std() if len(user_data) > 1 else 0,
            'weekend_activity_ratio': user_data['is_weekend'].mean(),
            'most_active_weekday': user_data['day_of_week'].mode().iloc[0] if not user_data['day_of_week'].empty else 0
        })
        
        # ميزات مشتقة (مع تجنب القسمة على صفر)
        if unique_sessions > 0:
            features.update({
                'events_per_session': total_events / unique_sessions,
                'songs_per_session': songs_count / unique_sessions,
                'listening_time_per_session': features['total_listening_time'] / unique_sessions
            })
        else:
            features.update({
                'events_per_session': 0,
                'songs_per_session': 0,
                'listening_time_per_session': 0
            })
        
        if active_days > 0:
            features.update({
                'events_per_day': total_events / active_days,
                'songs_per_day': songs_count / active_days
            })
        else:
            features.update({
                'events_per_day': 0,
                'songs_per_day': 0
            })
        
        # معدلات التفاعل
        total_interactions = thumbs_up + thumbs_down + add_playlist
        if songs_count > 0:
            features['interaction_rate'] = total_interactions / songs_count
        else:
            features['interaction_rate'] = 0
        
        if thumbs_up + thumbs_down > 0:
            features['positive_feedback_ratio'] = thumbs_up / (thumbs_up + thumbs_down)
        else:
            features['positive_feedback_ratio'] = 0.5
        
        # معدلات أخرى
        if total_events > 0:
            features['error_rate'] = features['error_count'] / total_events
            features['help_rate'] = features['help_visits'] / total_events
        else:
            features['error_rate'] = 0
            features['help_rate'] = 0
        
        # التنظيف النهائي - معالجة القيم الشاذة
        for key, value in features.items():
            if key == 'userId':  # تخطي المعرف
                continue
            
            # التأكد من أن القيمة رقمية وليست null أو inf
            if isinstance(value, (int, float)):
                if pd.isna(value) or np.isinf(value) or value < 0:
                    features[key] = 0
                elif value > 1e6:  # قيم كبيرة جداً
                    features[key] = 1e6
        
        return features