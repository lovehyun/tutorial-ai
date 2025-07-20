import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# 모델 및 스케일러 불러오기
model = load_model("model_lstm_seq10.h5")
scaler = joblib.load("scaler_seq10.pkl")  # 저장한 정규화 객체 불러오기

# 데이터 로드
data = np.load("lstm_data.npz")
X = data["X"]
y = (data["y"] != 0).astype(int)

# 정규화
n_samples, seq_len, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)
X_scaled = scaler.transform(X_reshaped)
X = X_scaled.reshape(n_samples, seq_len, n_features)

# 데이터 분할
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 추론
y_pred = (model.predict(X_test) > 0.5).astype(int)

# 평가 출력
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


import numpy as np

# 기존 데이터 모양 참조
n_samples, seq_len, n_features = X.shape

# 랜덤 샘플 생성 (정규화 전 기준)
random_sample = np.random.rand(seq_len, n_features)  # shape: (10, feature_dim)

# 정규화 (기존 학습된 스케일러 사용)
random_sample_reshaped = random_sample.reshape(-1, n_features)
random_sample_scaled = scaler.transform(random_sample_reshaped)
random_sample_scaled = random_sample_scaled.reshape(1, seq_len, n_features)

# 추론
y_prob = model.predict(random_sample_scaled)[0][0]
y_pred = int(y_prob > 0.5)

print(f"예측 결과: 클래스 {y_pred} (확률={y_prob:.4f})")
