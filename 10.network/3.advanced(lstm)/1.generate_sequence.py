# 1_generate_sequence.py
#  시간 정보(timestamp)를 추가하여 시계열 형태의 네트워크 트래픽 데이터로 바꾸는 코드

import pandas as pd
import numpy as np

df = pd.read_csv("network_multiclass.csv")

# 시간순 정렬을 흉내내기 위해 랜덤 셔플 제거
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 무작위 셔플 제거	.sample() 호출로 순서를 랜덤하게 배치 (시간 순서를 흉내내기 위해)
# timestamp 열 생성	1초 간격으로 가상의 시간 생성 (2023-01-01 00:00:00부터 시작)

# 가상의 timestamp 추가
df["timestamp"] = pd.date_range("2023-01-01", periods=len(df), freq="s")

# 저장
df.to_csv("network_sequence.csv", index=False)
