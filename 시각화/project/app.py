# app.py
import streamlit as st
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="폐업 위험 예측 대시보드", layout="wide")
st.title("🏪 가맹점 폐업 위험 예측 (Cox 비례위험모형)")

# -------------------------------
# 1️⃣ 데이터 불러오기 및 전처리
# -------------------------------
st.header("1️⃣ 데이터 불러오기 및 전처리")

file_path = "Total_Data_v2.CSV"  # CSV 파일은 app.py와 같은 폴더에 위치
try:
    df = pd.read_csv(file_path)
    st.success(f"'{file_path}' 파일 불러오기 성공")
except Exception as e:
    st.error(f"파일 불러오기 실패: {e}")
    st.stop()

# 날짜 컬럼 변환
for col in ['개설일', '폐업일', '기준년월']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    else:
        st.error(f"필수 컬럼 '{col}'가 데이터에 없습니다.")
        st.stop()

df['기준년월'] = df['기준년월'].dt.to_period('M').dt.to_timestamp()
df['폐업일'] = df['폐업일'].fillna(df['기준년월'])

# 운영개월, event 계산
df['운영개월'] = ((df['폐업일'].dt.year - df['개설일'].dt.year) * 12 +
                 (df['폐업일'].dt.month - df['개설일'].dt.month)).clip(lower=0)
df['event'] = np.where(df['폐업일'].notna() & (df['폐업일'] < df['기준년월']), 1, 0)

# 텍스트 컬럼 제거
drop_cols = ['가맹점명', '가맹점주소']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

st.write("📊 데이터 미리보기:", df.head())
st.write(f"데이터셋 크기: {df.shape}")

# -------------------------------
# 2️⃣ 변수 선택 및 정제
# -------------------------------
st.header("2️⃣ 변수 선택 및 정제")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['event', '운영개월']]

# 분산 낮은 컬럼 제거
low_var_cols = [col for col in numeric_cols if df[col].var() < 1e-5]
if low_var_cols:
    st.warning(f"분산 낮은 컬럼 제거: {low_var_cols}")
    df = df.drop(columns=low_var_cols)
    numeric_cols = [col for col in numeric_cols if col not in low_var_cols]

# 결측치/무한대 처리
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['운영개월', 'event'] + numeric_cols)

# VIF 계산
def calculate_vif(df, cols):
    vif_data = pd.DataFrame()
    vif_data["feature"] = cols
    vif_data["VIF"] = [variance_inflation_factor(df[cols].values, i)
                       for i in range(len(cols))]
    return vif_data

vif_df = calculate_vif(df, numeric_cols)
st.write("📉 VIF 계산 결과:", vif_df)

high_vif_cols = vif_df[vif_df["VIF"] > 10]["feature"].tolist()
if high_vif_cols:
    st.warning(f"다중공선성 높은 컬럼 제거: {high_vif_cols}")
    df = df.drop(columns=high_vif_cols)
    numeric_cols = [col for col in numeric_cols if col not in high_vif_cols]

# -------------------------------
# 3️⃣ Cox 비례위험모형 적합
# -------------------------------
st.header("3️⃣ Cox 비례위험모형 적합 결과")

cph = CoxPHFitter()
cph.fit(df[['운영개월', 'event'] + numeric_cols],
        duration_col='운영개월', event_col='event')

summary_df = cph.summary.reset_index()
summary_df.rename(columns={summary_df.columns[0]: '변수명'}, inplace=True)

st.write("📋 Cox 회귀계수 요약:", summary_df)

# Concordance Index
c_index = concordance_index(df['운영개월'], -cph.predict_partial_hazard(df), df['event'])
st.metric(label="📈 Concordance Index", value=f"{c_index:.4f}")

# -------------------------------
# 4️⃣ 변수별 위험도 시각화
# -------------------------------
st.header("4️⃣ 변수별 위험도 시각화")

fig, ax = plt.subplots(figsize=(8, len(summary_df) * 0.4))
sns.barplot(x='coef', y='변수명', data=summary_df, ax=ax, palette='coolwarm')
ax.axvline(0, color='black', linestyle='--')
ax.set_title("Cox 회귀계수 (양수: 위험 증가, 음수: 위험 감소)")
st.pyplot(fig)

# -------------------------------
# 5️⃣ 예시 생존 곡선
# -------------------------------
st.header("5️⃣ 예시 생존곡선 (전체 평균 기준)")

# Cox 모델 학습에 사용한 숫자형 컬럼만 선택
cols_for_mean = numeric_cols
df_numeric_mean = df[cols_for_mean].mean().to_frame().T  # 1행 dataframe

# 생존 함수 예측
surv = cph.predict_survival_function(df_numeric_mean)
st.line_chart(surv.T)
