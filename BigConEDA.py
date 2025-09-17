import pandas as pd

data1 = pd.read_csv("C:/Users/sht03/Desktop/2025_빅콘테스트_데이터_레이아웃_20250902/1456289/big_data_set1_f.csv", encoding='cp949')

data1['ARE_D'] = pd.to_datetime(data1['ARE_D'])
data1['MCT_ME_D'] = pd.to_datetime(data1['MCT_ME_D'])
counts = data1["HPSN_MCT_ZCD_NM"].value_counts()
pd.set_option("display.max_rows", None)  # 모든 행 보이게
print(counts)
print(counts)
print('hello')