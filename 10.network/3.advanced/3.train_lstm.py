# 3_train_lstm.py

# pip install numpy pandas scikit-learn tensorflow

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 데이터 로드
data = np.load("lstm_data.npz")
X = data["X"]
y = data["y"]

# 이진 분류로 제한 (0: 정상 vs 1/2: 이상)
y = (y != 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 모델 정의
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
model.save("model_lstm_seq10.h5")

# 평가
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
