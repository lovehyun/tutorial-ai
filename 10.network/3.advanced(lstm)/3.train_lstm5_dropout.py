# Dropout을 추가하여 과적합 방지

import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드
data = np.load("lstm_data.npz")
X = data["X"]
y = (data["y"] != 0).astype(int)

# 2. 정규화
n_samples, seq_len, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(n_samples, seq_len, n_features)

# 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. 모델 구성 (Dropout 추가)
model = Sequential()
model.add(LSTM(64, input_shape=(seq_len, n_features), return_sequences=False))
model.add(Dropout(0.3))  # 과적합 방지
model.add(Dense(1, activation='sigmoid'))

# 5. 학습
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 6. 모델 저장
model.save("model_lstm_dropout_seq10.h5")
joblib.dump(scaler, "scaler_dropout_seq10.pkl")

# 7. 평가
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
