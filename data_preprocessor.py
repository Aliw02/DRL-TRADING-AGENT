# data_preprocessor.py (Updated)
import pandas as pd
import joblib
import numpy as np
from config.init import config

# تحديد مسار الـ Scaler النهائي
FINAL_SCALER_PATH = "results/final_model_for_live/final_robust_scaler.joblib"
FEATURE_COLUMNS = config.get('data.feature_columns', [])

def load_scaler():
    """
    يقوم بتحميل الـ RobustScaler الذي تم حفظه.
    """
    try:
        scaler = joblib.load(FINAL_SCALER_PATH)
        print("تم تحميل الـ Scaler بنجاح.")
        return scaler
    except FileNotFoundError:
        print(f"خطأ: ملف الـ Scaler غير موجود في المسار: {FINAL_SCALER_PATH}")
        return None

def preprocess_data(df, scaler, current_position=None):
    """
    يقوم بتجهيز البيانات الجديدة لتكون جاهزة للموديل.
    الآن تستقبل موقف التداول الحالي كمُدخل.
    """
    if df is None or scaler is None:
        return None
        
    try:
        # التأكد من وجود الأعمدة المطلوبة
        features_df = df[FEATURE_COLUMNS].copy()
        
        # تحويل البيانات باستخدام الـ Scaler
        scaled_features = scaler.transform(features_df)
        
        # إضافة بعد ثالث (sequence_length) للبيانات
        sequence_length = config.get('environment.sequence_length', 40)
        
        # التأكد من وجود عدد كافٍ من الصفوف
        if len(scaled_features) < sequence_length:
            print("there is not enough data.")
            return None

        # أخذ آخر 'sequence_length' من البيانات
        sequence = scaled_features[-sequence_length:]
        
        # إضافة خاصية الموقف (Position) التي تدرب عليها النموذج
        # الآن نستخدم الموقف الحقيقي بدلاً من الافتراضي
        position_feature = np.zeros((sequence_length, 1))
        if current_position is not None:
            position_feature.fill(1)
        
        # دمج كل الخصائص
        processed_data = np.concatenate([sequence, position_feature], axis=1)
        
        # إعادة تشكيل البيانات لتناسب النموذج
        # النموذج يتوقع (1, sequence_length, features_dim + 1)
        return np.expand_dims(processed_data, axis=0)

    except Exception as e:
        print(f"there is an error: {e}")
        return None