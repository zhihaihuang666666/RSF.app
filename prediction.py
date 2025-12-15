# 导入包
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

## ===================== 加载模型 =====================##
#加载模型
#model = joblib.load("C:/Users/HZH/Desktop/生存机器学习模型/streamlit.app/RSF/survrf_model.pkl")
model = joblib.load("survrf_model.pkl")
# 获取模型特征
FEATURES = model.feature_names_in_

# 特征配置
CATEGORICAL_FEATURES = ["Hypertension", "Memory problem", "Gender"]
FEATURE_NAMES = { 
    "Hypertension": "Hypertension",
    "Memory problem": "Memory problem",
    "Age": "Age(years)",
    "Gender": "Gender",
    "Weight": "Weight(kg)",
    "WC": "Waist circumference(cm)",
    "HDL-C": "HDL-C(mg/dL)",
    "FBG": "FBG(mg/dL)",
    "HbA1c": "HbA1c(%)"

}

## ===================== Streamlit 页面配置 =====================##
st.set_page_config(page_title="CMM Prediction Model", layout="wide", initial_sidebar_state="expanded")
st.title("🫀 CMM Prediction Model")

## ===================== 用户输入界面 =====================##
input_data = {} 
col1, col2 = st.columns(2)
for i, feature in enumerate(FEATURES):
    with col1 if i % 2 == 0 else col2:
        feature_name = FEATURE_NAMES.get(feature, feature)
        if feature in CATEGORICAL_FEATURES:
            if feature == "Gender":
                val = st.selectbox(
                    f"{feature_name}",
                    options=[0, 1],
                    format_func=lambda x: "Male" if x == 1 else "Female",
                    key=feature,
                    index=1  # 将默认值设置为1（Male）
                )
            else:
                val = st.selectbox(
                    f"{feature_name}",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    key=feature,
                    index=1  # 将默认值设置为1（Yes）
                )
        else:
            if feature == "Age":
                val = st.number_input(f"{feature_name}", min_value=50.0, max_value=150.0, value=60.0, step=1.0)
            elif feature == "Weight":
                val = st.number_input(f"{feature_name}", min_value=20.0, max_value=200.0, value=60.0, step=0.1)
            elif feature == "WC":
                val = st.number_input(f"{feature_name}", min_value=20.0, max_value=150.0, value=80.0, step=0.1)         
            elif feature == "FBG":
                val = st.number_input(f"{feature_name}", min_value=50.0, max_value=200.0, value=110.0, step=0.1) 
            elif feature == "HbA1c":
                val = st.number_input(f"{feature_name}", min_value=3.0, max_value=20.0, value=5.0, step=0.1)
            elif feature == "HDL-C":
                val = st.number_input(f"{feature_name}", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
        input_data[feature] = val

## ===================== 预测逻辑 =====================##
if st.button("Predict CMM"):
    try:
        # 准备输入数据
        df_input = pd.DataFrame([input_data], columns=FEATURES)
        
        # 处理分类特征
        for col in df_input.columns:
            if df_input[col].dtype == object:
                le = LabelEncoder()
                df_input[col] = le.fit_transform(df_input[col].astype(str))
        
        # 预测生存函数
        survival_function = model.predict_survival_function(df_input)[0]
        # 设置时间点（2,4,7,9年）
        time_points = [2, 4, 7, 9]
        
        # 显示累积发病率结果（1 - 生存概率）
        st.subheader("📊 Cumulative incidence probability")
        for years in time_points:
            # 计算累积发病率：1 - 生存概率
            survival_prob = survival_function(years)
            cumulative_incidence = 1 - survival_prob
            st.write(f"**{years}-year incidence probability:** {cumulative_incidence:.1%}")
        
        # 显示累积发病率曲线
        st.subheader("📈 Time to incidence")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 计算累积发病率曲线：1 - 生存函数
        cumulative_incidence_curve = 1 - survival_function.y
        
        # 绘制累积发病率曲线
        ax.plot(survival_function.x, cumulative_incidence_curve, linewidth=1.5, color='#00A3FE')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Cumulative incidence probability')
        ax.set_title('Cumulative Incidence Curve')
        ax.grid(True, alpha=0.3)
        
        # 标记关键时间点
        for years in time_points:
            prob = 1 - survival_function(years)
            ax.plot(years, prob, 'ro', markersize=3) #ro红色圆点
            ax.annotate(f'{prob:.1%}', (years, prob), 
                       xytext=(years+0.5, prob+0.05), 
                       arrowprops=dict(arrowstyle='->'))
        
        # 设置y轴范围为0到1
        ax.set_ylim(0, 1)
        st.pyplot(fig,use_container_width=True)
        
    except Exception as e:
        st.error(f"预测过程出错: {str(e)}")


## 打开终端win+R,再运行streamlit run "C:/Users/HZH/Desktop/生存机器学习模型/streamlit.app/RSF/prediction.py"##



