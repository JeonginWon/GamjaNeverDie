import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, recall_score, 
    precision_score, f1_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚨 가맹점 위기 조기 경보 시스템 - 완전판")
print("   데이터 분석 → 모델 학습 → 성능 검증 → 점수화 시스템")
print("="*70)

# ========================
# 1. 데이터 로드 및 전처리
# ========================
print("\n[Step 1] 데이터 로드 및 전처리")

def load_csv_auto_encoding(filepath):
    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1', 'iso-8859-1']
    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except:
            continue
    raise Exception(f"파일을 읽을 수 없습니다: {filepath}")

df1 = load_csv_auto_encoding("C:/Users/SSAFY/Desktop/big_data_set1_f.csv")
df2 = load_csv_auto_encoding("C:/Users/SSAFY/Desktop/big_data_set2_f.csv")
df3 = load_csv_auto_encoding("C:/Users/SSAFY/Desktop/big_data_set3_f.csv")

df_time = df2.merge(df3, on=['ENCODED_MCT', 'TA_YM'], how='inner')
df_all = df_time.merge(df1, on='ENCODED_MCT', how='inner')

df_all['기준년월'] = pd.to_datetime(df_all['TA_YM'], format='%Y%m')
df_all['폐업일'] = pd.to_datetime(df_all['MCT_ME_D'], format='%Y%m%d', errors='coerce')

def create_target(row):
    if pd.isna(row['폐업일']):
        return 0
    months_until_closure = (row['폐업일'].year - row['기준년월'].year) * 12 + \
                          (row['폐업일'].month - row['기준년월'].month)
    return 1 if 0 < months_until_closure <= 12 else 0

df_all['위기신호'] = df_all.apply(create_target, axis=1)

# 이상값 처리
for col in ['DLV_SAA_RAT', 'M1_SME_RY_SAA_RAT', 'M12_SME_BZN_ME_MCT_RAT']:
    if col in df_all.columns:
        df_all.loc[df_all[col] < -999, col] = np.nan

# 업종 대분류
업종_대분류_매핑 = {
    '한식': ['한식-육류/고기', '한식-단품요리일반', '한식-해물/생선', '한식-국수/만두', 
             '한식-국밥/설렁탕', '한식-찌개/전골', '백반/가정식', '한정식', '한식뷔페', 
             '한식-죽', '한식-두부요리', '한식-냉면', '한식-감자탕'],
    '중식': ['중식당', '중식-딤섬/중식만두', '중식-훠궈/마라탕'],
    '일식': ['일식당', '일식-초밥/롤', '일식-덮밥/돈가스', '일식-우동/소바/라면',
             '일식-샤브샤브', '일식-참치회', '이자카야'],
    '양식': ['양식', '스테이크', '피자', '햄버거', '샌드위치/토스트'],
    '카페/디저트': ['카페', '커피전문점', '테이크아웃커피', '테마카페', '베이커리', 
                   '도너츠', '아이스크림/빙수', '마카롱', '와플/크로플', '차', '주스', 
                   '떡/한과', '떡/한과 제조', '탕후루'],
    '치킨/분식': ['치킨', '분식', '꼬치구이'],
    '주점/술': ['호프/맥주', '요리주점', '민속주점', '와인바', '룸살롱/단란주점', 
                '일반 유흥주점', '포장마차'],
    '식재료/판매': ['축산물', '수산물', '농산물', '청과물', '미곡상', '건어물', 
                   '식료품', '유제품', '주류', '와인샵', '건강식품', '건강원', 
                   '인삼제품', '담배', '식품 제조']
}

업종_to_대분류 = {}
for 대분류, 소분류들 in 업종_대분류_매핑.items():
    for 소분류 in 소분류들:
        업종_to_대분류[소분류] = 대분류

df_all['업종_대분류'] = df_all['HPSN_MCT_ZCD_NM'].fillna('미분류').map(
    lambda x: 업종_to_대분류.get(x, '기타')
)

le_industry = LabelEncoder()
le_location = LabelEncoder()
df_all['업종_인코딩'] = le_industry.fit_transform(df_all['업종_대분류'])
df_all['지역_인코딩'] = le_location.fit_transform(df_all['HPSN_MCT_BZN_CD_NM'].fillna('미분류'))

def parse_range(value):
    if pd.isna(value) or (isinstance(value, (int, float)) and value < 0):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    value = str(value)
    if '%' in value:
        parts = value.split('_')
        if len(parts) > 1:
            range_part = parts[1]
            if '미만' in range_part:
                return float(range_part.replace('%미만', '').replace('%', '')) / 2
            elif '-' in range_part:
                nums = range_part.replace('%', '').split('-')
                return (float(nums[0]) + float(nums[1])) / 2
    return np.nan

for col in ['MCT_OPE_MS_CN', 'RC_M1_SAA', 'RC_M1_TO_UE_CT', 'RC_M1_UE_CUS_CN', 'RC_M1_AV_NP_AT']:
    if col in df_all.columns:
        df_all[col + '_숫자'] = df_all[col].apply(parse_range)

features = [
    '업종_인코딩', '지역_인코딩',
    'MCT_OPE_MS_CN_숫자', 'RC_M1_SAA_숫자', 'RC_M1_TO_UE_CT_숫자',
    'RC_M1_UE_CUS_CN_숫자', 'RC_M1_AV_NP_AT_숫자',
    'DLV_SAA_RAT', 'M1_SME_RY_SAA_RAT', 'M1_SME_RY_CNT_RAT',
    'M12_SME_RY_SAA_PCE_RT', 'M12_SME_BZN_SAA_PCE_RT',
    'M12_SME_RY_ME_MCT_RAT', 'M12_SME_BZN_ME_MCT_RAT',
    'MCT_UE_CLN_REU_RAT', 'MCT_UE_CLN_NEW_RAT',
    'RC_M1_SHC_RSD_UE_CLN_RAT', 'RC_M1_SHC_WP_UE_CLN_RAT', 'RC_M1_SHC_FLP_UE_CLN_RAT',
    'M12_MAL_30_RAT', 'M12_FME_30_RAT'
]

available_features = [f for f in features if f in df_all.columns]

for feat in available_features:
    if df_all[feat].isna().sum() > 0:
        median_val = df_all[feat].median()
        df_all[feat] = df_all[feat].fillna(median_val if not pd.isna(median_val) else 0)

df_clean = df_all[available_features + ['위기신호', '기준년월', 'ENCODED_MCT', '업종_대분류']].copy()

print(f"  ✓ 전처리 완료: {len(df_clean):,}개 레코드")

# ========================
# 2. Train/Test 분할
# ========================
print("\n[Step 2] Train/Test 분할 (시간 기반)")

SPLIT_DATE = '2024-04-01'
train_df = df_clean[df_clean['기준년월'] < SPLIT_DATE]
test_df = df_clean[df_clean['기준년월'] >= SPLIT_DATE]

X_train = train_df[available_features]
y_train = train_df['위기신호']
X_test = test_df[available_features]
y_test = test_df['위기신호']

print(f"  Train: {len(X_train):,}개 (위기: {y_train.sum()}개, {y_train.mean()*100:.2f}%)")
print(f"  Test:  {len(X_test):,}개 (위기: {y_test.sum()}개, {y_test.mean()*100:.2f}%)")

# ========================
# 3. 스케일링 및 SMOTE
# ========================
print("\n[Step 3] 스케일링 및 오버샘플링")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"  Before SMOTE: {len(X_train):,}개 (위기: {y_train.sum()}개)")
print(f"  After SMOTE:  {len(X_train_balanced):,}개 (위기: {y_train_balanced.sum()}개)")

# ========================
# 4. 다중 모델 학습
# ========================
print("\n[Step 4] 다중 모델 학습")

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced'),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, max_depth=5, eval_metric='logloss')
}

trained_models = {}
predictions = {}

for name, model in models.items():
    print(f"  학습 중: {name}...", end=" ")
    model.fit(X_train_balanced, y_train_balanced)
    trained_models[name] = model
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    predictions[name] = y_pred_proba
    
    print("✓")

# ========================
# 5. 모델 성능 평가
# ========================
print("\n[Step 5] 모델 성능 평가")

print("\n  5-1. 기본 성능 (AUC)")
print(f"  {'Model':<20} {'AUC':<10}")
print(f"  {'-'*30}")

auc_scores = {}
for name, y_pred_proba in predictions.items():
    auc = roc_auc_score(y_test, y_pred_proba)
    auc_scores[name] = auc
    print(f"  {name:<20} {auc:.4f}")

print("\n  5-2. 임계값 최적화 (F1 Score 기준)")
print(f"  {'Model':<20} {'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1':<10}")
print(f"  {'-'*70}")

best_results = {}

for name, y_pred_proba in predictions.items():
    # F1 최대화 임계값 찾기
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # 최적 임계값으로 예측
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    best_results[name] = {
        'threshold': best_threshold,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'auc': auc_scores[name],
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"  {name:<20} {best_threshold:>11.4f} {recall:>9.4f} {precision:>11.4f} {f1:>9.4f}")

# 최고 성능 모델 선택
best_model_name = max(best_results.items(), key=lambda x: x[1]['f1'])[0]
best_model = trained_models[best_model_name]
best_result = best_results[best_model_name]

print(f"\n  🏆 최고 성능 모델: {best_model_name}")
print(f"     F1 Score: {best_result['f1']:.4f}")
print(f"     AUC: {best_result['auc']:.4f}")
print(f"     Recall: {best_result['recall']:.4f}")
print(f"     Precision: {best_result['precision']:.4f}")

# ========================
# 6. 상세 성능 분석
# ========================
print("\n[Step 6] 상세 성능 분석 (최고 모델 기준)")

y_pred_best = best_result['y_pred']

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm.ravel()

print(f"\n  혼동 행렬:")
print(f"  {'':<15} {'예측: 안전':<15} {'예측: 위기'}")
print(f"  {'실제: 안전':<15} {tn:<15} {fp}")
print(f"  {'실제: 위기':<15} {fn:<15} {tp}")

print(f"\n  실전 해석 (Test 데이터 {len(y_test):,}개):")
print(f"  • 실제 위기: {y_test.sum()}개")
print(f"    ├─ 모델이 탐지: {tp}개 ({tp/y_test.sum()*100:.1f}%)")
print(f"    └─ 모델이 놓침: {fn}개 ({fn/y_test.sum()*100:.1f}%)")
print(f"  • 모델 경보 발송: {tp+fp}개")
print(f"    ├─ 실제 위기: {tp}개 (정확도: {tp/(tp+fp)*100:.1f}%)")
print(f"    └─ 오경보: {fp}개 ({fp/(tp+fp)*100:.1f}%)")

# Classification Report
print(f"\n  분류 보고서:")
print(classification_report(y_test, y_pred_best, target_names=['안전', '위기']))

# ========================
# 7. Feature Importance (최고 모델)
# ========================
print("\n[Step 7] 변수 중요도 분석")

if best_model_name in ['RandomForest', 'GradientBoosting', 'XGBoost']:
    importances = best_model.feature_importances_
elif best_model_name == 'LogisticRegression':
    importances = np.abs(best_model.coef_[0])

importance_df = pd.DataFrame({
    '변수': available_features,
    '중요도': importances
}).sort_values('중요도', ascending=False)

print(f"\n  변수 중요도 TOP 15 ({best_model_name}):")
for idx, row in importance_df.head(15).iterrows():
    bar = "█" * int(row['중요도'] * 100)
    print(f"  {row['변수']:<30} {row['중요도']:>8.4f} {bar}")

# 업종 변수 순위
if '업종_인코딩' in importance_df['변수'].values:
    업종_순위 = importance_df[importance_df['변수'] == '업종_인코딩'].index[0] + 1
    업종_중요도 = importance_df[importance_df['변수'] == '업종_인코딩']['중요도'].values[0]
    print(f"\n  💡 업종 변수: {업종_순위}위 (중요도: {업종_중요도:.4f})")

# ========================
# 8. 임계값별 성능 비교
# ========================
print("\n[Step 8] 임계값별 성능 비교 (최고 모델)")

print(f"\n  {'Threshold':<12} {'경보수':<10} {'실제위기':<10} {'Precision':<12} {'Recall':<10} {'F1'}")
print(f"  {'-'*70}")

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    y_pred_temp = (best_result['y_pred_proba'] >= threshold).astype(int)
    
    num_alerts = y_pred_temp.sum()
    true_crisis = (y_test[y_pred_temp == 1]).sum()
    
    if num_alerts > 0:
        prec = true_crisis / num_alerts
        rec = true_crisis / y_test.sum()
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    else:
        prec = rec = f1 = 0
    
    print(f"  {threshold:<12.2f} {num_alerts:<10} {true_crisis:<10} {prec*100:>10.1f}% {rec*100:>9.1f}% {f1:>9.4f}")

# ========================
# 9. 최종 결과 정리
# ========================
print("\n" + "="*70)
print("📊 최종 모델 성능 요약")
print("="*70)

print(f"\n데이터셋:")
print(f"  • Train: {len(X_train):,}개 (위기: {y_train.sum()}개)")
print(f"  • Test: {len(X_test):,}개 (위기: {y_test.sum()}개)")

print(f"\n모델 비교:")
for name in ['LogisticRegression', 'RandomForest', 'GradientBoosting', 'XGBoost']:
    result = best_results[name]
    print(f"  {name:<20} AUC: {result['auc']:.4f}  F1: {result['f1']:.4f}  Recall: {result['recall']:.4f}")

print(f"\n🏆 최종 선택 모델: {best_model_name}")
print(f"  • AUC: {best_result['auc']:.4f}")
print(f"  • F1 Score: {best_result['f1']:.4f}")
print(f"  • Recall: {best_result['recall']*100:.1f}% (위기의 {best_result['recall']*100:.1f}% 탐지)")
print(f"  • Precision: {best_result['precision']*100:.1f}% (경보의 {best_result['precision']*100:.1f}%가 실제 위기)")
print(f"  • 최적 임계값: {best_result['threshold']:.4f}")

print(f"\n실전 활용:")
print(f"  • {len(y_test):,}개 가맹점 중 {y_test.sum()}개 위기 발생")
print(f"  • 모델이 {tp}개 사전 탐지 ({tp/y_test.sum()*100:.1f}%)")
print(f"  • {fp}개 오경보 발생")
print(f"  • 조기 개입 가능 가맹점: {tp}개")

print("\n" + "="*70)
print("✅ 모델 학습 및 검증 완료!")
print("="*70)

# 모델 저장 (선택사항)
print("\n[선택] 모델 저장")
save_model = input("학습된 모델을 저장하시겠습니까? (y/n): ")
if save_model.lower() == 'y':
    import pickle
    
    with open(f'{best_model_name}_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"  ✓ 모델 저장 완료: {best_model_name}_model.pkl")
    print(f"  ✓ 스케일러 저장 완료: scaler.pkl")