import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# 1. 页面配置与标题
# ==========================================
st.set_page_config(page_title="骨髓转移智能预测系统", layout="wide")

st.title("🧬 骨髓转移风险预测与SHAP可解释性分析工具")
st.markdown("""
本工具基于机器学习模型预测恶性肿瘤患者发生 **骨髓转移 (Bone Marrow Metastasis)** 的风险，
并利用 **SHAP** 算法解释各临床指标对预测结果的影响。
***
""")

# ==========================================
# 2. 模拟模型训练 (实际项目中请加载训练好的模型)
# ==========================================
@st.cache_resource # 缓存模型，避免每次刷新都重练
def train_demo_model():
    # 模拟 500 个患者数据
    np.random.seed(42)
    n_samples = 500
    data = pd.DataFrame({
        'Age': np.random.randint(20, 85, n_samples),
        'LDH (U/L)': np.random.normal(250, 100, n_samples), # 乳酸脱氢酶
        'ALP (U/L)': np.random.normal(120, 60, n_samples),  # 碱性磷酸酶
        'Hemoglobin (g/L)': np.random.normal(110, 20, n_samples), # 血红蛋白
        'Platelet (10^9/L)': np.random.normal(200, 80, n_samples), # 血小板
        'Primary_Lung': np.random.randint(0, 2, n_samples), # 原发灶: 肺
        'Primary_Breast': np.random.randint(0, 2, n_samples) # 原发灶: 乳腺
    })
    
    # 模拟标签：LDH高、ALP高、Hb低 容易转移
    risk = (data['LDH (U/L)'] * 0.02 + data['ALP (U/L)'] * 0.01 - 
            data['Hemoglobin (g/L)'] * 0.05 + np.random.normal(0, 2, n_samples))
    labels = (risk > risk.mean()).astype(int)
    
    # 训练模型
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    return model, X_train

model, X_train = train_demo_model()

# ==========================================
# 3. 侧边栏：用户输入临床特征
# ==========================================
st.sidebar.header("📋 患者临床特征输入")

def user_input_features():
    age = st.sidebar.slider('年龄 (Age)', 18, 90, 55)
    ldh = st.sidebar.number_input('乳酸脱氢酶 (LDH, U/L)', min_value=50.0, max_value=2000.0, value=250.0)
    alp = st.sidebar.number_input('碱性磷酸酶 (ALP, U/L)', min_value=30.0, max_value=1000.0, value=120.0)
    hb = st.sidebar.number_input('血红蛋白 (Hemoglobin, g/L)', min_value=30.0, max_value=200.0, value=110.0)
    plt_count = st.sidebar.number_input('血小板 (Platelet, 10^9/L)', min_value=10.0, max_value=600.0, value=200.0)
    
    primary_cancer = st.sidebar.selectbox('原发肿瘤部位', ('肺癌', '乳腺癌', '其他'))
    
    # 转换为模型需要的格式
    primary_lung = 1 if primary_cancer == '肺癌' else 0
    primary_breast = 1 if primary_cancer == '乳腺癌' else 0
    
    input_df = pd.DataFrame({
        'Age': [age],
        'LDH (U/L)': [ldh],
        'ALP (U/L)': [alp],
        'Hemoglobin (g/L)': [hb],
        'Platelet (10^9/L)': [plt_count],
        'Primary_Lung': [primary_lung],
        'Primary_Breast': [primary_breast]
    })
    return input_df

input_df = user_input_features()

# ==========================================
# 4. 主界面：预测与SHAP解释
# ==========================================

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 输入概览")
    st.dataframe(input_df.T.style.format("{:.1f}"))
    
    predict_btn = st.button('开始预测分析', type='primary')

if predict_btn:
    # --- A. 预测结果 ---
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    
    st.markdown("---")
    st.subheader("🎯 预测结果")
    
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        if probability > 0.5:
            st.error(f"高风险 (Positive)")
        else:
            st.success(f"低风险 (Negative)")
            
    with metric_col2:
        st.metric(label="骨髓转移概率", value=f"{probability:.2%}")
    
    st.progress(float(probability))

    # --- B. SHAP 解释 ---
    st.markdown("---")
    st.subheader("🔍 SHAP 可解释性分析")
    st.info("下图展示了各特征如何推动预测结果：红色表示增加风险，蓝色表示降低风险。")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)
    
    # 1. 瀑布图 (针对单个样本最直观的解释)
    st.write("**1. 局部解释：单样本瀑布图 (Waterfall Plot)**")
    fig1, ax1 = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False, max_display=7)
    st.pyplot(fig1, bbox_inches='tight')
    
    # 2. 力导向图 (传统视图)
    st.write("**2. 局部解释：力导向图 (Force Plot)**")
    # Force plot 需要 javascript 支持，使用 streamlit components 渲染
    try:
        import streamlit.components.v1 as components
        shap.initjs()
        force_plot = shap.force_plot(explainer.expected_value, shap_values.values[0], input_df, matplotlib=False)
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        components.html(shap_html, height=150)
    except:
        st.warning("交互式 Force Plot 渲染失败，请查看静态瀑布图。")

    # 3. 全局特征重要性 (可选，帮助医生理解模型整体逻辑)
    with st.expander("查看模型全局特征重要性 (Summary Plot)"):
        st.write("基于训练集的整体特征影响分布：")
        # 需要计算训练集的 shap values，比较耗时，Demo中仅计算少量
        shap_values_train = explainer(X_train.iloc[:100])
        fig2, ax2 = plt.subplots()
        shap.plots.beeswarm(shap_values_train, show=False)
        st.pyplot(fig2, bbox_inches='tight')

# ==========================================
# 5. 免责声明
# ==========================================
st.markdown("---")
st.caption("⚠️ 免责声明：本工具仅供医学科研与辅助参考，不能替代医生的专业临床诊断。预测结果请结合患者实际临床表现综合判断。")