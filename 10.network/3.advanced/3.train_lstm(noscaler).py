# 3_train_lstm.py

# pip install numpy pandas scikit-learn tensorflow

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. 데이터 로드
data = np.load("lstm_data.npz")
# 이전에 준비한 시계열 데이터셋 불러옴
# X: 10초 시퀀스 데이터 (4개의 특성 포함)
# y: 다음 시점의 라벨 (0: 정상, 1/2: 이상)
X = data["X"] # (samples, 10, 4)
y = data["y"] # (samples,)


# 다중 클래스(0/1/2)를 이진 클래스(0=정상, 1=이상)로 변환
# → 이진 분류용 LSTM 훈련

# 이진 분류로 제한 (0: 정상 vs 1/2: 이상)
y = (y != 0).astype(int)

# 2. train/test 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. LSTM 모델 정의
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

# 4. 모델 컴파일 및 저장
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 5. 모델 저장
model.save("model_lstm_seq10.h5")

# 6. 평가
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
