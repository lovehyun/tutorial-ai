# 목표: 기존 network_sequence.csv 데이터에서 **이전 10초 동안의 네트워크 특성(feature)**을 보고,
# **다음 시간의 라벨(label)**을 예측하기 위한 **LSTM 학습 데이터셋(X, y)**을 생성합니다.

# 2_prepare_lstm_data.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 설정
sequence_length = 10  # 최근 10초의 데이터를 보고 다음 상태 예측

# CSV 파일 로드 및 시간순 정렬
df = pd.read_csv("network_sequence.csv")
df = df.sort_values("timestamp")

# 레이블과 입력 분리
features = ["duration", "packet_size", "src_bytes", "dst_bytes"]
label_col = "label"

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# 시퀀스 데이터 생성
X = []
y = []

for i in range(len(df) - sequence_length):
    seq_x = df[features].iloc[i:i+sequence_length].values
    seq_y = df[label_col].iloc[i+sequence_length]
    X.append(seq_x)
    y.append(seq_y)

X = np.array(X)
y = np.array(y)

# X: 이전 10개의 특성 시퀀스 (seq_x)
# y: 그 다음 시점의 클래스 라벨 (seq_y)
# → 결과적으로 (samples, 10, 4) 형태의 X와 (samples,) 형태의 y가 생성됨

# 훈련/테스트 분리
np.savez("lstm_data.npz", X=X, y=y)


# 기본 실험 (sequence_length = 10)용 데이터 생성
# → GRU, Dropout, Multiclass 등에서도 공유해서 사용 가능
