# 1_generate_sequence.py

import pandas as pd
import numpy as np

df = pd.read_csv("network_multiclass.csv")

# 시간순 정렬을 흉내내기 위해 랜덤 셔플 제거
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 가상의 timestamp 추가
df["timestamp"] = pd.date_range("2023-01-01", periods=len(df), freq="S")

# 저장
df.to_csv("network_sequence.csv", index=False)
