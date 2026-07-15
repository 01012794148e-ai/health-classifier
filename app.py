import streamlit as st
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import joblib


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("classification_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"خطأ في تحميل ملفات النموذج: {e}")

# 2. تصميم واجهة المستخدم
st.title("تطبيق تشخيص الحالات (Healthy vs Sick)")
st.write("قم برفع ملف `.mat` يحتوي على البيانات لاستخراج التوقعات.")

# مكان لرفع الملف
uploaded_file = st.file_uploader("اختر ملف .mat", type=["mat"])

if uploaded_file is not None:
    try:
        # قراءة ملف الـ mat
        mat_data = loadmat(uploaded_file)
        
        # البحث عن اسم المصفوفة داخل الملف تلقائياً
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        if len(keys) == 0:
            st.error("الملف فارغ أو لا يحتوي على مصفوفات صالحة.")
        else:
            # اختيار أول مصفوفة بيانات متاحة بالملف
            feature_name = keys[0]
            data_matrix = mat_data[feature_name]
            
            st.info(f"تم العثور على المصفوفة: `{feature_name}` بحجم {data_matrix.shape}")
            
            # التأكد من مطابقة عدد الميزات (52 ميزة)
            if data_matrix.shape[1] != 52 and data_matrix.shape[0] == 52:
                data_matrix = data_matrix.T
                
            if data_matrix.shape[1] == 52:
                # 3. معالجة البيانات وعمل التوقع
                scaled_data = scaler.transform(data_matrix)
                predictions = model.predict(scaled_data)
                
                st.subheader("نتائج التوقع:")
                
                # عرض النتيجة لكل عينة
                for i, pred in enumerate(predictions):
                    prob = float(pred[0])
                    # 1 تعني healthy و 0 تعني sick
                    status = "Healthy (سليم)" if prob >= 0.5 else "Sick (مريض)"
                    confidence = prob if prob >= 0.5 else (1 - prob)
                    
                    st.write(f"العينة رقم {i+1}: **{status}** (نسبة التأكد: {confidence*100:.2f}%)")
            else:
                st.error(f"خطأ: يجب أن يحتوي الملف على 52 ميزة (Features). الحجم الحالي: {data_matrix.shape}")
                
    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الملف: {e}")
