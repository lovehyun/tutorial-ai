# GRU(Gated Recurrent Unit)로 변경
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드
data = np.load("lstm_data.npz")
X = data["X"]
y = (data["y"] != 0).astype(int)

# 2. 정규화 (입력: 3D → 2D → 3D)
n_samples, seq_len, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)  # (samples * seq_len, features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(n_samples, seq_len, n_features)

# 3. 학습용/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. GRU 모델 구성
model = Sequential()
model.add(GRU(64, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1, activation='sigmoid'))

# 5. 학습 및 저장
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
model.save("model_gru_seq10.h5")
joblib.dump(scaler, "scaler_gru_seq10.pkl")  # 나중에 추론용

# 6. 평가
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
