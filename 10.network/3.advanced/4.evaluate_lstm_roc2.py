import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


# 1. 데이터 로드
data = np.load("lstm_data.npz")
X = data["X"]
y = (data["y"] != 0).astype(int)  # 0: 정상, 1: 이상

# 2. 정규화 (LSTM은 정규화된 입력이 중요함)
n_samples, seq_len, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(n_samples, seq_len, n_features)

# 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. 모델 구성
model = Sequential()
model.add(LSTM(64, input_shape=(seq_len, n_features)))
model.add(Dense(1, activation='sigmoid'))

# 5. 학습
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 6. 모델 및 스케일러 저장
model.save("model_lstm_seq10_with_scaler.h5")
joblib.dump(scaler, "scaler_seq10.pkl")

# 7. 평가: ROC Curve
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob > 0.5).astype(int)

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. 시각화
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve (LSTM 이진 분류)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
