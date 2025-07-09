# 2_prepare_lstm_data.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 설정
sequence_length = 10  # 최근 10초의 데이터를 보고 다음 상태 예측

df = pd.read_csv("network_sequence.csv")
df = df.sort_values("timestamp")  # 시간순 정렬

# 레이블과 입력 분리
features = ["duration", "packet_size", "src_bytes", "dst_bytes"]
label_col = "label"

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# 시퀀스 생성
X = []
y = []

for i in range(len(df) - sequence_length):
    seq_x = df[features].iloc[i:i+sequence_length].values
    seq_y = df[label_col].iloc[i+sequence_length]
    X.append(seq_x)
    y.append(seq_y)

X = np.array(X)
y = np.array(y)

# 훈련/테스트 분리
np.savez("lstm_data.npz", X=X, y=y)


# 기본 실험 (sequence_length = 10)용 데이터 생성
# → GRU, Dropout, Multiclass 등에서도 공유해서 사용 가능
