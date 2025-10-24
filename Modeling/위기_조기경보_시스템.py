import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚨 가맹점 위기 조기 경보 시스템")
print("   AI 기반 위험 신호 탐지 및 맞춤형 경보")
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

# 데이터 로드
df1 = load_csv_auto_encoding("C:/Users/SSAFY/Desktop/big_data_set1_f.csv")
df2 = load_csv_auto_encoding("C:/Users/SSAFY/Desktop/big_data_set2_f.csv")
df3 = load_csv_auto_encoding("C:/Users/SSAFY/Desktop/big_data_set3_f.csv")

# 병합
df_time = df2.merge(df3, on=['ENCODED_MCT', 'TA_YM'], how='inner')
df_all = df_time.merge(df1, on='ENCODED_MCT', how='inner')

# 날짜 변환
df_all['기준년월'] = pd.to_datetime(df_all['TA_YM'], format='%Y%m')
df_all['폐업일'] = pd.to_datetime(df_all['MCT_ME_D'], format='%Y%m%d', errors='coerce')

# 타깃 변수 (위기 신호)
def create_crisis_target(row):
    """위기 신호: 12개월 내 폐업 예정"""
    if pd.isna(row['폐업일']):
        return 0
    months_until_closure = (row['폐업일'].year - row['기준년월'].year) * 12 + \
                          (row['폐업일'].month - row['기준년월'].month)
    return 1 if 0 < months_until_closure <= 12 else 0

df_all['위기신호'] = df_all.apply(create_crisis_target, axis=1)

# 이상값 처리
for col in ['DLV_SAA_RAT', 'M1_SME_RY_SAA_RAT', 'M12_SME_BZN_ME_MCT_RAT']:
    if col in df_all.columns:
        df_all.loc[df_all[col] < -999, col] = np.nan

# 업종 대분류 매핑
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

# 구간 변수 변환
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

for col in ['MCT_OPE_MS_CN', 'RC_M1_SAA', 'RC_M1_TO_UE_CT']:
    if col in df_all.columns:
        df_all[col + '_숫자'] = df_all[col].apply(parse_range)

# 핵심 변수 선택
핵심변수 = [
    'DLV_SAA_RAT',  # 배달매출비율
    'M1_SME_RY_SAA_RAT',  # 동일업종 매출비율
    'M1_SME_RY_CNT_RAT',  # 동일업종 건수비율
    'M12_SME_RY_SAA_PCE_RT',  # 동일업종 순위
    'M12_SME_BZN_ME_MCT_RAT',  # 상권 폐업비중
    'MCT_OPE_MS_CN_숫자',  # 운영개월수
    'RC_M1_TO_UE_CT_숫자',  # 매출건수
]

# 결측치 처리
for var in 핵심변수:
    if var in df_all.columns and df_all[var].isna().sum() > 0:
        median_val = df_all[var].median()
        df_all[var] = df_all[var].fillna(median_val if not pd.isna(median_val) else 0)

df_clean = df_all[핵심변수 + ['위기신호', '기준년월', 'ENCODED_MCT', '업종_대분류']].copy()

print(f"  ✓ 전처리 완료: {len(df_clean):,}개 레코드")
print(f"  ✓ 위기 신호: {df_clean['위기신호'].sum():,}개")

# ========================
# 2. 위험 신호 발견 (EDA)
# ========================
print("\n[Step 2] 데이터에서 위험 신호 발견")

위기_그룹 = df_clean[df_clean['위기신호'] == 1]
안전_그룹 = df_clean[df_clean['위기신호'] == 0]

print(f"\n  발견된 주요 위험 신호:")

위험신호_발견 = []
for var in 핵심변수:
    위기_평균 = 위기_그룹[var].mean()
    안전_평균 = 안전_그룹[var].mean()
    차이 = 위기_평균 - 안전_평균
    
    if 안전_평균 != 0:
        차이율 = (차이 / 안전_평균) * 100
    else:
        차이율 = 0
    
    # 통계 검정
    위기_데이터 = 위기_그룹[var].dropna()
    안전_데이터 = 안전_그룹[var].dropna()
    
    if len(위기_데이터) > 0 and len(안전_데이터) > 0:
        t_stat, p_value = stats.ttest_ind(위기_데이터, 안전_데이터)
        
        위험신호_발견.append({
            '변수': var,
            '위기_평균': 위기_평균,
            '안전_평균': 안전_평균,
            '차이': 차이,
            '차이율(%)': 차이율,
            'p_value': p_value,
            '유의함': p_value < 0.05
        })

위험신호_df = pd.DataFrame(위험신호_발견).sort_values('차이율(%)', key=abs, ascending=False)

for idx, row in 위험신호_df.head(5).iterrows():
    direction = "↑ 높음" if row['차이'] > 0 else "↓ 낮음"
    significance = "***" if row['유의함'] else ""
    print(f"  • {row['변수']:<25} {direction} ({abs(row['차이율(%)']):.1f}%) {significance}")

# ========================
# 3. 위험 점수화 시스템 구축
# ========================
print("\n[Step 3] 위험 점수화 시스템 구축")

# 위험 규칙 생성
위험규칙 = {}
for idx, row in 위험신호_df.iterrows():
    var = row['변수']
    
    위험규칙[var] = {
        'p25': df_clean[var].quantile(0.25),
        'p50': df_clean[var].quantile(0.50),
        'p75': df_clean[var].quantile(0.75),
        '위험방향': 'low' if row['차이'] < 0 else 'high',
        '가중치': abs(row['차이율(%)']) / 100  # 차이율 기반 가중치
    }

print(f"\n  위험 점수 계산 규칙:")
for var, rule in 위험규칙.items():
    방향 = "낮을수록" if rule['위험방향'] == 'low' else "높을수록"
    print(f"  • {var:<25} {방향:8} 위험 (가중치: {rule['가중치']:.2f})")

def calculate_risk_score(row, 위험규칙):
    """
    가맹점의 위험 점수 계산 (0~100점)
    """
    total_score = 0
    total_weight = sum(rule['가중치'] for rule in 위험규칙.values())
    
    for var, rule in 위험규칙.items():
        value = row[var]
        
        if pd.isna(value):
            continue
        
        # 위험 점수 계산
        if rule['위험방향'] == 'low':
            # 낮을수록 위험
            if value <= rule['p25']:
                score = 100
            elif value <= rule['p50']:
                score = 70
            elif value <= rule['p75']:
                score = 30
            else:
                score = 0
        else:
            # 높을수록 위험
            if value >= rule['p75']:
                score = 100
            elif value >= rule['p50']:
                score = 70
            elif value >= rule['p25']:
                score = 30
            else:
                score = 0
        
        # 가중치 적용
        total_score += score * rule['가중치']
    
    return total_score / total_weight if total_weight > 0 else 0

def classify_risk_level(score):
    """위험 점수를 등급으로 분류"""
    if score >= 70:
        return '🔴 위험'
    elif score >= 50:
        return '🟠 주의'
    elif score >= 30:
        return '🟡 관심'
    else:
        return '🟢 안전'

# ========================
# 4. 전체 데이터에 적용
# ========================
print("\n[Step 4] 위험 점수 산출")

df_clean['위험점수'] = df_clean.apply(
    lambda row: calculate_risk_score(row, 위험규칙), axis=1
)
df_clean['위험등급'] = df_clean['위험점수'].apply(classify_risk_level)

print(f"\n  위험 점수 통계:")
print(f"  • 평균: {df_clean['위험점수'].mean():.1f}점")
print(f"  • 중앙값: {df_clean['위험점수'].median():.1f}점")
print(f"  • 범위: {df_clean['위험점수'].min():.1f} ~ {df_clean['위험점수'].max():.1f}점")

print(f"\n  위험 등급 분포:")
for level, count in df_clean['위험등급'].value_counts().sort_index().items():
    pct = count / len(df_clean) * 100
    print(f"  {level}: {count:,}개 ({pct:.1f}%)")

# 등급별 실제 위기 발생률
print(f"\n  등급별 실제 위기 발생률:")
등급별_위기율 = df_clean.groupby('위험등급')['위기신호'].agg(['sum', 'count', 'mean'])
등급별_위기율['발생률(%)'] = 등급별_위기율['mean'] * 100
print(등급별_위기율[['sum', 'count', '발생률(%)']].to_string())

# ========================
# 5. 업종별 분석
# ========================
print("\n[Step 5] 업종별 위험 분석")

업종별_분석 = df_clean.groupby('업종_대분류').agg({
    'ENCODED_MCT': 'nunique',
    '위기신호': ['sum', 'mean'],
    '위험점수': 'mean'
}).round(2)

업종별_분석.columns = ['가맹점수', '위기건수', '위기율', '평균위험점수']
업종별_분석['위기율(%)'] = 업종별_분석['위기율'] * 100
업종별_분석 = 업종별_분석.sort_values('위기율', ascending=False)

print("\n  업종별 위기율:")
for 업종, row in 업종별_분석.iterrows():
    bar = "█" * int(row['위기율(%)'] * 10)
    print(f"  {업종:<15} {row['위기율(%)']:>5.2f}% {bar} (평균점수: {row['평균위험점수']:.1f})")

# ========================
# 6. 개별 가맹점 리포트 생성 함수
# ========================
print("\n[Step 6] 개별 가맹점 리포트 생성 기능")

def generate_merchant_report(merchant_id, reference_date=None):
    """
    개별 가맹점의 위험 리포트 생성
    """
    # 해당 가맹점 데이터 추출
    if reference_date:
        merchant_data = df_clean[
            (df_clean['ENCODED_MCT'] == merchant_id) & 
            (df_clean['기준년월'] == reference_date)
        ]
    else:
        # 최신 데이터
        merchant_data = df_clean[df_clean['ENCODED_MCT'] == merchant_id].sort_values('기준년월', ascending=False).head(1)
    
    if len(merchant_data) == 0:
        return {"error": "가맹점을 찾을 수 없습니다"}
    
    row = merchant_data.iloc[0]
    
    # 위험 점수 및 등급
    위험점수 = row['위험점수']
    위험등급 = row['위험등급']
    업종 = row['업종_대분류']
    
    # 업종 평균과 비교
    업종_평균점수 = 업종별_분석.loc[업종, '평균위험점수']
    
    # 주요 위험 요인 분석
    위험요인 = []
    for var, rule in 위험규칙.items():
        value = row[var]
        
        if pd.isna(value):
            continue
        
        # 위험도 판단
        if rule['위험방향'] == 'low':
            if value <= rule['p25']:
                위험도 = "높음"
                상태 = f"하위 25% ({value:.1f})"
            elif value <= rule['p50']:
                위험도 = "보통"
                상태 = f"하위 50% ({value:.1f})"
            else:
                위험도 = "낮음"
                상태 = f"양호 ({value:.1f})"
        else:
            if value >= rule['p75']:
                위험도 = "높음"
                상태 = f"상위 25% ({value:.1f})"
            elif value >= rule['p50']:
                위험도 = "보통"
                상태 = f"상위 50% ({value:.1f})"
            else:
                위험도 = "낮음"
                상태 = f"양호 ({value:.1f})"
        
        위험요인.append({
            "지표": var,
            "위험도": 위험도,
            "상태": 상태
        })
    
    # 위험도별 정렬
    위험도_순서 = {"높음": 0, "보통": 1, "낮음": 2}
    위험요인.sort(key=lambda x: 위험도_순서[x["위험도"]])
    
    # 조치 사항
    if 위험등급 == '🔴 위험':
        조치사항 = [
            "🚨 긴급 상담 필요: 소상공인지원센터 연락 (☎ 1357)",
            "💰 긴급 운영자금 지원 신청 검토",
            "📋 경영 개선 컨설팅 신청"
        ]
    elif 위험등급 == '🟠 주의':
        조치사항 = [
            "📊 월간 매출 모니터링 강화",
            "🎯 주요 위험 지표 개선 계획 수립",
            "💡 온라인 마케팅 교육 수강"
        ]
    elif 위험등급 == '🟡 관심':
        조치사항 = [
            "✅ 분기별 경영 점검",
            "👥 고객 피드백 수집",
            "🔍 경쟁 가맹점 모니터링"
        ]
    else:
        조치사항 = [
            "✨ 현재 안정적 운영 중",
            "📈 성장 전략 수립 권장",
            "🎁 단골 고객 보상 프로그램 검토"
        ]
    
    return {
        "가맹점ID": merchant_id,
        "업종": 업종,
        "위험점수": round(위험점수, 1),
        "위험등급": 위험등급,
        "업종평균": round(업종_평균점수, 1),
        "주요위험요인": 위험요인[:5],  # 상위 5개
        "조치사항": 조치사항
    }

# 샘플 리포트 생성
print("\n  샘플 가맹점 리포트:")
sample_merchant = df_clean[df_clean['위험등급'] == '🔴 위험']['ENCODED_MCT'].iloc[0]
sample_report = generate_merchant_report(sample_merchant)

print(f"\n  {'='*60}")
print(f"  가맹점 위험 리포트")
print(f"  {'='*60}")
print(f"  가맹점 ID: {sample_report['가맹점ID']}")
print(f"  업종: {sample_report['업종']}")
print(f"  위험 점수: {sample_report['위험점수']}점 (업종 평균: {sample_report['업종평균']}점)")
print(f"  위험 등급: {sample_report['위험등급']}")
print(f"\n  주요 위험 요인:")
for 요인 in sample_report['주요위험요인']:
    print(f"    • {요인['지표']:<25} {요인['위험도']:4} - {요인['상태']}")
print(f"\n  권장 조치사항:")
for idx, 조치 in enumerate(sample_report['조치사항'], 1):
    print(f"    {idx}. {조치}")

print("\n" + "="*70)
print("✅ 조기 경보 시스템 구축 완료!")
print("="*70)

print("\n💡 시스템 활용 방법:")
print("  1. generate_merchant_report(가맹점ID) 함수로 개별 리포트 생성")
print("  2. 위험점수 70점 이상 가맹점에 조기 경보 발송")
print("  3. 업종별 맞춤 지원 프로그램 운영")
print("  4. 월간 모니터링으로 위험도 변화 추적")

# 결과 저장 (선택사항)
print("\n[선택] 결과 저장")
save_option = input("결과를 CSV로 저장하시겠습니까? (y/n): ")
if save_option.lower() == 'y':
    output_file = 'merchant_risk_scores.csv'
    df_clean[['ENCODED_MCT', '기준년월', '업종_대분류', '위험점수', '위험등급', '위기신호']].to_csv(
        output_file, index=False, encoding='utf-8-sig'
    )
    print(f"  ✓ 저장 완료: {output_file}")