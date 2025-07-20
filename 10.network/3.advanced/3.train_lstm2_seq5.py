# sequence_length = 5일 때의 LSTM 이진 분류

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# 1. 데이터 로드
data = np.load("lstm_data_seq5.npz")
X = data["X"]  # shape = (samples, 5, 4)
y = (data["y"] != 0).astype(int)  # 0: 정상, 1/2: 이상 → 1로 통합

# 2. 정규화 (StandardScaler는 2D 입력 필요)
n_samples, seq_len, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)  # (samples * 5, 4)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)

X = X_scaled.reshape(n_samples, seq_len, n_features)

# 3. train/test 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. 모델 정의
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1, activation='sigmoid'))

# 5. 모델 학습
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 6. 모델 저장
model.save("model_lstm_seq5.h5")
joblib.dump(scaler, "scaler_seq5.pkl")

# 7. 평가
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
