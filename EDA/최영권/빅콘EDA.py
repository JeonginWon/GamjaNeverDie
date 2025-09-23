import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data1 = pd.read_csv("C:/Users/sht03/Desktop/2025_빅콘테스트_데이터_레이아웃_20250902/1456289/big_data_set1_f.csv", encoding='cp949')
data2 = pd.read_csv("C:/Users/sht03/Desktop/2025_빅콘테스트_데이터_레이아웃_20250902/1456289/big_data_set2_f.csv", encoding='cp949')

# data1['ARE_D'] = pd.to_datetime(data1['ARE_D'])
# data1['MCT_ME_D'] = pd.to_datetime(data1['MCT_ME_D'])
# counts = data1["HPSN_MCT_ZCD_NM"].value_counts()
# pd.set_option("display.max_rows", None)  # 모든 행 보이게
# print(counts)
# print(counts)

# print(data2[data2['APV_CE_RAT'].notnull()].value_counts())
data2_filtered = data2[data2["APV_CE_RAT"].notna()]
cols_front = ["MCT_OPE_MS_CN", "RC_M1_SAA", "RC_M1_TO_UE_CT", "RC_M1_UE_CUS_CN", "RC_M1_AV_NP_AT", "APV_CE_RAT"]
for col in cols_front:
    data2_filtered[col] = data2_filtered[col].str.split("_").str[0].astype(int)
# print(data2_filtered[["MCT_OPE_MS_CN", "RC_M1_SAA", "RC_M1_TO_UE_CT", "RC_M1_UE_CUS_CN", "RC_M1_AV_NP_AT", "APV_CE_RAT"]])

# 상관계수 계산
corr = data2_filtered[cols_front].corr(method="spearman")

# 히트맵 시각화 -> "MCT_OPE_MS_CN" ~ "RC_M1_AV_NP_AT"까지 상관관계
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Spearman Correlation Heatmap Front")
# plt.show()


# cols_back = ["DLV_SAA_RAT", "M1_SME_RY_SAA_RAT", "M1_SME_RY_CNT_RAT", "M12_SME_RY_SAA_PCE_RT", "M12_SME_BZN_SAA_PCE_RT", "M12_SME_RY_ME_MCT_RAT", "M12_SME_BZN_ME_MCT_RAT"]
# for col in cols_back:
#     data2_filtered[col] = data2_filtered[col].std.split("_").str[0].astype(int)
# # print(data2_filtered[["MCT_OPE_MS_CN", "RC_M1_SAA", "RC_M1_TO_UE_CT", "RC_M1_UE_CUS_CN", "RC_M1_AV_NP_AT", "APV_CE_RAT"]])

# # 상관계수 계산
# corr = data2_filtered[cols_back].corr(method="spearman")

# # 히트맵 시각화 -> "MCT_OPE_MS_CN" ~ "RC_M1_AV_NP_AT"까지 상관관계
# plt.figure(figsize=(8,6))
# sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
# plt.title("Spearman Correlation Heatmap Back")
# plt.show()

# DLV_SAA_RAT -999999.9 여부에 따라 그룹 나누기
group_invalid = data2_filtered[data2_filtered["DLV_SAA_RAT"] == -999999.9]
group_valid = data2_filtered[data2_filtered["DLV_SAA_RAT"] != -999999.9]

# 그룹별 개수 확인
len_invalid = len(group_invalid)
len_valid = len(group_valid)

(len_invalid, len_valid)

# 총 유효 데이터 개수
total_valid = len(group_valid)

# 3개 그룹으로 최대한 비슷하게 나누기
size = total_valid // 3
remainder = total_valid % 3

# 앞의 그룹부터 나머지를 하나씩 추가
sizes = [size + (1 if i < remainder else 0) for i in range(3)]

# 그룹 분할
group1_balanced = group_valid.iloc[:sizes[0]]
group2_balanced = group_valid.iloc[sizes[0]:sizes[0]+sizes[1]]
group3_balanced = group_valid.iloc[sizes[0]+sizes[1]:]

print(len(group1_balanced), len(group2_balanced), len(group3_balanced))
