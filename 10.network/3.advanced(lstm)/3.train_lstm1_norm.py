# 정규화가 필요한 이유
# LSTM 같은 딥러닝 모델은 입력 특성들의 스케일에 매우 민감합니다. 예를 들어:
# feature	    예시 값
# duration	    20
# packet_size	200
# src_bytes	    500
# dst_bytes	    600
#
# 이렇게 스케일이 제각각일 경우, 큰 값에만 민감하게 반응하고 작은 값은 무시하게 됩니다.
# 그래서 StandardScaler나 MinMaxScaler 등을 통해 모든 특성의 분포를 맞춰줘야 학습이 잘 됩니다.

# 3_train_lstm.py

# pip install numpy pandas scikit-learn tensorflow

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. 데이터 로드
data = np.load("lstm_data.npz")
# X: 10초 시퀀스 데이터 (4개의 특성 포함)
# y: 다음 시점의 라벨 (0: 정상, 1/2: 이상)
X = data["X"] # (samples, 10, 4)
y = data["y"] # (samples,)


# 2. 이전에 준비한 시계열 데이터셋 불러옴

# 다중 클래스(0/1/2)를 이진 클래스(0=정상, 1=이상)로 변환
# → 이진 분류용 LSTM 훈련

# 이진 분류로 제한 (0: 정상 vs 1/2: 이상)
y = (y != 0).astype(int)

# 3. 정규화 (StandardScaler 적용)
# → LSTM에 들어가기 전에 특성별 스케일을 맞춰줌
scaler = StandardScaler()

# (samples, 10, 4) → (samples * 10, 4)
X_reshaped = X.reshape(-1, X.shape[2])  # (전체 시계열 수 * 길이, 특성)
X_scaled = scaler.fit_transform(X_reshaped)

# 다시 원래 차원으로 복원
X = X_scaled.reshape(X.shape)

# 저장: 추론 시 동일한 scaler 사용
joblib.dump(scaler, "scaler_seq10.pkl")

# 4. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. LSTM 모델 정의
# 입력: (10초, 4개 특성)
# LSTM(64) → 시퀀스 특징을 학습 :
#  - 입력인자: (배치 크기, 시퀀스 길이, 특성 수) = (batch_size, timesteps, features)
#  - X.shape(): 예: (1490, 10, 4)
#    X.shape[0] = 1490	샘플 개수 (전체 시퀀스 수)
#    X.shape[1] = 10	시퀀스 길이 (10초)
#    X.shape[2] = 4     각 시점의 특성 수 (duration, packet_size, src_bytes, dst_bytes)
# Dense(1, sigmoid) → 확률 형태의 이진 출력 (정상 vs 이상)
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1, activation='sigmoid'))

# 6. 모델 컴파일 및 저장
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 7. 모델 저장
model.save("model_lstm_seq10.h5")

# 8. 평가
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# 하이퍼파라미터 실험
# 파라미터	설명	추천 실험값
# batch_size	배치 크기	16, 32, 64
# epochs	학습 반복 수	5, 10, 20
# optimizer	최적화 알고리즘	adam, rmsprop, sgd
# loss	손실 함수	binary_crossentropy, categorical_crossentropy
# dropout	과적합 방지	0.2 ~ 0.5

# 실험 종류별 예시 조합
# 실험 ID	sequence_length	모델 구조	출력 형태	목적
# A1	5	LSTM(64)	sigmoid	단기 예측
# A2	15	LSTM(128→64)	sigmoid	장기 예측
# B1	10	LSTM + Dropout	sigmoid	과적합 방지
# B2	10	GRU(64)	sigmoid	구조 변화
# C1	10	LSTM(64)	softmax	다중 클래스
