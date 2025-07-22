# 다중 클래스 분류 (label = 0, 1, 2 직접 예측)
# 시계열 기반 LSTM 모델을 사용하여 네트워크 트래픽 데이터를 3개의 클래스(정상, DoS, Probe)로 
# 분류하는 다중 클래스 분류 모델을 학습하고 평가하는 전체 파이프라인

import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 1. 데이터 로드
data = np.load("lstm_data.npz")
X = data["X"]
y = data["y"].astype(int)

# 2. 정규화
n_samples, seq_len, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(n_samples, seq_len, n_features)

# 3. 다중 클래스 변환
y_cat = to_categorical(y, num_classes=3)

# 4. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y, random_state=42
)

# 5. 모델 구성
model = Sequential()
model.add(LSTM(64, input_shape=(seq_len, n_features)))
model.add(Dense(3, activation='softmax'))

# 6. 학습
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 7. 저장
model.save("model_lstm_multiclass_seq10.h5")
joblib.dump(scaler, "scaler_multiclass_seq10.pkl")

# 8. 평가
y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
