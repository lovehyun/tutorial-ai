import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

model = load_model("model_lstm_seq10.h5")

# 패턴 정의: 0은 정상, 1은 공격
pattern = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # 4번째만 공격

X_manual = []
for p in pattern:
    if p == 0:
        # 정상 트래픽 특징
        features = [
            np.random.normal(20, 4),
            np.random.normal(200, 20),
            np.random.normal(500, 50),
            np.random.normal(600, 50),
        ]
    else:
        # 공격 트래픽 특징 (DoS 등)
        features = [
            np.random.normal(2, 1),
            np.random.normal(1500, 100),
            np.random.normal(3000, 300),
            np.random.normal(100, 30),
        ]
    X_manual.append(features)

X_manual = np.array(X_manual)  # shape = (10, 4)

# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_manual)
X_input = np.expand_dims(X_scaled, axis=0)  # (1, 10, 4)

# 예측
y_prob = model.predict(X_input)[0][0]
y_pred = int(y_prob > 0.5)

print(f"입력 패턴: {pattern}")
print(f"예측 결과: {'이상' if y_pred else '정상'} (확률: {y_prob:.3f})")
