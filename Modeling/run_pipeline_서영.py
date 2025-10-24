# final_auc_model.py
# 가맹점 폐업 예측 모델 - AUC 0.6 목표 (완전판)

import os
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIG
# =========================
CSV_PATH = Path(r"C:/Users/SSAFY/Desktop/Total_Data_v2.csv")
RANDOM_STATE = 42

CLOSURE_PREDICTION_MONTHS = 12  # 6개월은 샘플 부족! 12개월 유지
TIME_SPLIT_DATE = '2024-01-01'

# 목표
TARGET_AUC = 0.60

# 전략
USE_SMOTE = True
USE_ENSEMBLE = True  # Voting 앙상블

# 저장
OUT_DIR = Path("./results_final")
OUT_DIR.mkdir(exist_ok=True)

# 제거할 변수 (진짜 누수만)
LEAKAGE_FEATURES = [
    '폐업일', '개설일', '기준년월_최신', '가맹점구분번호',
]

# 사용할 변수 (추세 신호 포함)
SAFE_FEATURES = [
    '가맹점 운영개월수 구간', '배달매출금액 비율',
    '매출금액 구간', '매출건수 구간', '유니크 고객 수 구간',
    '객단가 구간', '취소율 구간',
    
    # 강력한 추세 신호
    '전월대비 매출금액 감소율(%)',
    '3개월 연속 감소 여부',
    '6개월 하락추세 여부',
    '급감여부(-20% YoY)',
    '매출 안정성(변동성) CV(3개월)',
    '매출 안정성(변동성) 여부',
    '매출금액 구간_회복지수_6개월',
    '매출탄력도(6개월)',
    
    # 상대 평가
    '동일 업종 매출금액 비율_x', '동일 업종 매출금액 비율_y',
    '동일 업종 매출건수 비율', '동일 업종 내 매출 순위 비율',
    '동일 상권 내 매출 순위 비율_x', '동일 상권 내 매출 순위 비율_y',
    '동일 업종 내 해지 가맹점 비중_x', '동일 업종 내 해지 가맹점 비중_y',
    '동일 상권 내 해지 가맹점 비중',
    
    # 고객 특성
    '남성 20대이하 고객 비중', '남성 30대 고객 비중', '남성 40대 고객 비중',
    '남성 50대 고객 비중', '남성 60대이상 고객 비중',
    '여성 20대이하 고객 비중', '여성 30대 고객 비중', '여성 40대 고객 비중',
    '여성 50대 고객 비중', '여성 60대이상 고객 비중',
    '재방문 고객 비중_x', '재방문 고객 비중_y', '신규 고객 비중',
    '거주 이용 고객 비율', '직장 이용 고객 비율', '유동인구 이용 고객 비율',
    '고객분포_다양성지수',
]

# =========================
# FUNCTIONS
# =========================
def load_and_parse_dates(path):
    """CSV 로드 및 날짜 파싱"""
    df = pd.read_csv(path, encoding='utf-8-sig', on_bad_lines='skip')
    
    df['기준년월_dt'] = pd.to_datetime(df['기준년월'], errors='coerce')
    df['폐업일_str'] = df['폐업일'].fillna(0).astype(int).astype(str).replace('0', None)
    df['폐업일_dt'] = pd.to_datetime(df['폐업일_str'], format='%Y%m%d', errors='coerce')
    
    n_closure = df[df['폐업일_dt'].notna()]['가맹점구분번호'].nunique()
    n_total = df['가맹점구분번호'].nunique()
    
    print(f"[INFO] 전체 가맹점: {n_total:,}개")
    print(f"       폐업 가맹점: {n_closure}개 ({n_closure/n_total*100:.1f}%)")
    
    return df


def create_store_level_dataset(df, months_ahead=12, cutoff_date='2024-01-01'):
    """가맹점 단위 데이터셋 생성"""
    df = df.copy()
    cutoff = pd.to_datetime(cutoff_date)
    
    df['폐업까지_개월'] = (
        (df['폐업일_dt'].dt.year - df['기준년월_dt'].dt.year) * 12 +
        (df['폐업일_dt'].dt.month - df['기준년월_dt'].dt.month)
    )
    
    store_records = []
    
    for store_id in df['가맹점구분번호'].unique():
        store_data = df[df['가맹점구분번호'] == store_id].sort_values('기준년월_dt')
        
        # Train: cutoff 이전 최신 데이터
        train_period = store_data[store_data['기준년월_dt'] < cutoff]
        if len(train_period) > 0:
            train_record = train_period.iloc[-1:].copy()
            if pd.notna(train_record['폐업일_dt'].iloc[0]):
                months = train_record['폐업까지_개월'].iloc[0]
                train_record['타깃'] = int((months > 0) and (months <= months_ahead))
            else:
                train_record['타깃'] = 0
            train_record['split'] = 'train'
            store_records.append(train_record)
        
        # Test: cutoff 이후 최신 데이터
        test_period = store_data[store_data['기준년월_dt'] >= cutoff]
        if len(test_period) > 0:
            test_record = test_period.iloc[-1:].copy()
            if pd.notna(test_record['폐업일_dt'].iloc[0]):
                months = test_record['폐업까지_개월'].iloc[0]
                test_record['타깃'] = int((months > 0) and (months <= months_ahead))
            else:
                test_record['타깃'] = 0
            test_record['split'] = 'test'
            store_records.append(test_record)
    
    result = pd.concat(store_records, ignore_index=True)
    
    train_df = result[result['split'] == 'train']
    test_df = result[result['split'] == 'test']
    
    print(f"\n[데이터셋 생성]")
    print(f"  Train: {len(train_df):,}개 (폐업={train_df['타깃'].sum()})")
    print(f"  Test: {len(test_df):,}개 (폐업={test_df['타깃'].sum()})")
    
    return result


def create_advanced_features(X):
    """고급 파생 변수 생성"""
    X = X.copy()
    
    # 1) Y/N → 0/1 변환
    yn_cols = {
        '3개월 연속 감소 여부': '3개월_연속감소',
        '6개월 하락추세 여부': '6개월_하락추세',
        '급감여부(-20% YoY)': '급감',
        '매출 안정성(변동성) 여부': '매출변동성'
    }
    
    for orig, new in yn_cols.items():
        if orig in X.columns:
            if X[orig].dtype == 'object':
                X[new] = X[orig].map({'Y': 1, 'N': 0, 'y': 1, 'n': 0}).fillna(0).astype(int)
            else:
                X[new] = (X[orig] == 1).astype(int)
    
    # 2) 위험 신호 개수
    signal_cols = ['3개월_연속감소', '6개월_하락추세', '급감', '매출변동성']
    available = [c for c in signal_cols if c in X.columns]
    if len(available) >= 2:
        X['위험신호_개수'] = X[available].sum(axis=1)
        X['위험신호_비율'] = X['위험신호_개수'] / len(available)
        X['다중위험'] = (X['위험신호_개수'] >= 2).astype(int)
    
    # 3) 매출 추세 점수
    if '전월대비 매출금액 감소율(%)' in X.columns:
        X['매출감소_위험도'] = np.clip(-X['전월대비 매출금액 감소율(%)'] / 50, 0, 1)
        X['극단적감소'] = (X['전월대비 매출금액 감소율(%)'] < -30).astype(int)
    
    # 4) 비율 변수 통계
    ratio_cols = [c for c in X.columns if '비율' in c or '비중' in c]
    if len(ratio_cols) > 5:
        X['비율_평균'] = X[ratio_cols].mean(axis=1)
        X['비율_표준편차'] = X[ratio_cols].std(axis=1)
    
    # 5) 순위
    rank_cols = [c for c in X.columns if '순위' in c]
    if len(rank_cols) > 1:
        X['순위_평균'] = X[rank_cols].mean(axis=1)
        X['순위_최악'] = X[rank_cols].max(axis=1)
    
    # 6) 주변 해지율
    closure_cols = [c for c in X.columns if '해지 가맹점 비중' in c]
    if len(closure_cols) > 1:
        X['주변_해지율_평균'] = X[closure_cols].mean(axis=1)
    
    # 7) 고객 다양성
    gender_age = [c for c in X.columns if ('남성' in c or '여성' in c) and '고객 비중' in c]
    if len(gender_age) > 5:
        X['고객_엔트로피'] = -1 * (X[gender_age] * np.log(X[gender_age] + 1e-10)).sum(axis=1)
    
    # 8) 복합 위험 점수
    risk_components = []
    if '위험신호_비율' in X.columns:
        risk_components.append(X['위험신호_비율'])
    if '주변_해지율_평균' in X.columns:
        max_val = X['주변_해지율_평균'].max()
        if max_val > 0:
            risk_components.append(X['주변_해지율_평균'] / max_val)
    if '순위_최악' in X.columns:
        risk_components.append(X['순위_최악'])
    
    if len(risk_components) > 0:
        X['복합위험점수'] = np.mean(risk_components, axis=0)
    
    return X


def preprocess_features(X):
    """특성 전처리"""
    X = X.copy()
    
    # 범주형 변수 처리
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        high_card = [c for c in cat_cols if X[c].nunique() > 50]
        if high_card:
            X = X.drop(columns=high_card)
            cat_cols = [c for c in cat_cols if c not in high_card]
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
    
    # 수치형 변환
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    
    # 고결측 제거 (50% 이상)
    high_miss = [c for c in X.columns if X[c].isna().mean() > 0.5]
    if high_miss:
        X = X.drop(columns=high_miss)
    
    # 이상치 처리 (IQR)
    for col in X.select_dtypes(include=[np.number]).columns:
        Q1, Q3 = X[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        X[col] = X[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    
    # 결측치 대체
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median() if pd.notna(X[c].median()) else 0)
    
    # 분산 0 제거
    zero_var = [c for c in X.columns if X[c].std() == 0]
    if zero_var:
        X = X.drop(columns=zero_var)
    
    # 파생 변수 생성
    X = create_advanced_features(X)
    
    return X


def main():
    print("="*70)
    print("🎯 가맹점 폐업 예측 모델 - AUC 0.6 목표")
    print("="*70)
    
    # ===== 1단계: 데이터 로드 =====
    print("\n[1단계] 데이터 로드")
    df = load_and_parse_dates(CSV_PATH)
    
    # ===== 2단계: 데이터셋 생성 =====
    print(f"\n[2단계] 데이터셋 생성 ({CLOSURE_PREDICTION_MONTHS}개월 예측)")
    df_store = create_store_level_dataset(df, CLOSURE_PREDICTION_MONTHS, TIME_SPLIT_DATE)
    
    # ===== 3단계: 전처리 =====
    print("\n[3단계] 특성 전처리")
    available = [c for c in SAFE_FEATURES if c in df_store.columns]
    X = df_store[available].copy()
    print(f"  원본 변수: {len(X.columns)}개")
    
    X = preprocess_features(X)
    print(f"  전처리 후: {len(X.columns)}개 (파생 변수 포함)")
    
    # Train/Test 분할
    train_mask = df_store['split'] == 'train'
    test_mask = df_store['split'] == 'test'
    
    X_train = X[train_mask].reset_index(drop=True)
    X_test = X[test_mask].reset_index(drop=True)
    y_train = df_store[train_mask]['타깃'].reset_index(drop=True)
    y_test = df_store[test_mask]['타깃'].reset_index(drop=True)
    
    if y_train.sum() < 5 or y_test.sum() < 5:
        print("\n⚠️  폐업 샘플 부족. 종료.")
        return
    
    # ===== 4단계: 스케일링 =====
    print("\n[4단계] 스케일링")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ===== 5단계: SMOTE =====
    if USE_SMOTE and y_train.sum() >= 6:
        print("\n[5단계] SMOTE")
        try:
            from imblearn.over_sampling import SMOTE
            k = min(5, y_train.sum() - 1)
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            print(f"  {len(X_train_scaled):,} → {len(X_train_resampled):,}")
        except Exception as e:
            print(f"  SMOTE 실패: {e}")
            X_train_resampled, y_train_resampled = X_train_scaled, y_train
    else:
        X_train_resampled, y_train_resampled = X_train_scaled, y_train
    
    # ===== 6단계: 모델 학습 =====
    print("\n[6단계] 모델 학습")
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    
    if USE_ENSEMBLE:
        print("  Voting 앙상블 구성...")
        
        # 개별 모델
        lr = LogisticRegression(
            max_iter=2000, 
            class_weight={0:1, 1:8}, 
            random_state=RANDOM_STATE
        )
        
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.03,
            random_state=RANDOM_STATE
        )
        
        estimators = [('lr', lr), ('rf', rf), ('gb', gb)]
        weights = [1, 2, 1.5]
        
        # XGBoost (있으면)
        try:
            from xgboost import XGBClassifier
            pos_w = len(y_train_resampled) / (y_train_resampled.sum() + 1)
            xgb = XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.03,
                scale_pos_weight=pos_w,
                random_state=RANDOM_STATE,
                eval_metric='logloss'
            )
            estimators.append(('xgb', xgb))
            weights.append(2)
            print("  XGBoost 포함")
        except:
            print("  XGBoost 제외")
        
        # LightGBM (있으면) - 불균형 데이터에 강함!
        try:
            from lightgbm import LGBMClassifier
            lgbm = LGBMClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.03,
                scale_pos_weight=50,  # 극단적 가중치
                random_state=RANDOM_STATE,
                verbose=-1
            )
            estimators.append(('lgbm', lgbm))
            weights.append(2.5)  # 가장 높은 가중치
            print("  LightGBM 포함")
        except:
            print("  LightGBM 제외 (pip install lightgbm)")
        
        # Voting
        model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )
        
        print("  학습 중...")
        model.fit(X_train_resampled, y_train_resampled)
        print("  완료!")
        
    else:
        # 단일 모델
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            class_weight='balanced_subsample',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train_resampled, y_train_resampled)
    
    # ===== 7단계: 평가 =====
    print("\n[7단계] 평가")
    from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix
    
    y_proba = model.predict_proba(X_test_scaled)[:,1]
    
    # 확률 분포 확인
    print(f"\n[확률 분포]")
    print(f"  Min: {y_proba.min():.4f}")
    print(f"  Max: {y_proba.max():.4f}")
    print(f"  Mean: {y_proba.mean():.4f}")
    print(f"  Median: {np.median(y_proba):.4f}")
    
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n{'='*70}")
    print(f"🎯 AUC: {auc:.4f} {'✅ 목표 달성!' if auc >= TARGET_AUC else f'(목표: {TARGET_AUC})'}")
    print(f"{'='*70}")
    
    # 임계값별 성능
    print(f"\n임계값별 성능:")
    print(f"{'임계값':<10} {'Recall':<10} {'Precision':<12} {'F1':<10}")
    print("-" * 45)
    
    for thresh in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
        y_pred = (y_proba >= thresh).astype(int)
        rec = recall_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        print(f"{thresh:<10.2f} {rec:<10.4f} {prec:<12.4f} {f1:<10.4f}")
    
    # 최적 F1
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, 0.05):
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    y_pred_best = (y_proba >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
    
    print(f"\n{'='*70}")
    print(f"최적 임계값: {best_thresh:.2f}")
    print(f"  Recall: {recall_score(y_test, y_pred_best):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_best, zero_division=0):.4f}")
    print(f"  F1: {best_f1:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"\n혼동 행렬:")
    print(f"  실제 폐업 & 탐지: {tp}개")
    print(f"  실제 폐업 & 놓침: {fn}개")
    print(f"  오경보: {fp}개")
    print(f"  정상 탐지: {tn}개")
    print(f"{'='*70}")
    
    # 저장
    results = pd.DataFrame([{
        'AUC': auc,
        'Best_Threshold': best_thresh,
        'Recall': recall_score(y_test, y_pred_best),
        'Precision': precision_score(y_test, y_pred_best, zero_division=0),
        'F1': best_f1,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn
    }])
    
    results.to_csv(OUT_DIR / "final_results.csv", index=False, encoding='utf-8-sig')
    pd.DataFrame({'feature': X.columns}).to_csv(OUT_DIR / "features.csv", index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 결과 저장: {OUT_DIR}")


if __name__ == "__main__":
    main()