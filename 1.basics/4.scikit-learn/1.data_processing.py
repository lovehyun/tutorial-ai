# 1_python_basics/4_data_processing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

print("데이터 전처리 실습")

# 결측값 포함 데이터
data = {'이름': ['철수', '영희', '민수', '지민'],
        '점수': [80, np.nan, 95, 88]}
df = pd.DataFrame(data)
print(df)

print("\n결측값 처리 (평균으로 대체)")
df['점수'] = df['점수'].fillna(df['점수'].mean())
print(df)

print("\n정규화")
scaler = MinMaxScaler()
df['점수_정규화'] = scaler.fit_transform(df[['점수']])
print(df)

print("\n라벨 인코딩")
labels = ['사과', '바나나', '오렌지', '바나나']
encoder = LabelEncoder()
encoded = encoder.fit_transform(labels)
print(f"라벨: {labels}")
print(f"인코딩 결과: {encoded}")
