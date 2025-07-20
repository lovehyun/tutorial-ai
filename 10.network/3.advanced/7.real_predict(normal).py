import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# 1. 랜덤으로 "정상"스러운 시퀀스 10개 만들기
# (실험 목적: 정상처럼 보이는 트래픽 예측)
sequence_length = 10
features = ["duration", "packet_size", "src_bytes", "dst_bytes"]

# 랜덤 시퀀스 생성 (정상 트래픽 기반 분포)
np.random.seed(42)
new_data = np.column_stack([
    np.random.normal(20, 4, sequence_length),      # duration
    np.random.normal(200, 20, sequence_length),    # packet_size
    np.random.normal(500, 50, sequence_length),    # src_bytes
    np.random.normal(600, 50, sequence_length),    # dst_bytes
])

# 2. 학습 때 저장해 둔 scaler 불러오기
scaler = joblib.load("scaler_seq10.pkl")  # ← 학습 시 저장했던 스케일러

# 3. 정규화
new_data_scaled = scaler.transform(new_data)  # (10, 4)
X_input = np.expand_dims(new_data_scaled, axis=0)  # (1, 10, 4)

# 4. 모델 불러오기
model = load_model("model_lstm_seq10.h5")

# 5. 추론
y_prob = model.predict(X_input)[0][0]
y_pred = int(y_prob > 0.5)

print(f"예측 결과: {'이상' if y_pred else '정상'} (확률: {y_prob:.3f})")


# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 214ms/step
# 예측 결과: 정상 (확률: 0.477)
# 확률이 0.477이므로, 0.5보다 낮아 정상(0) 클래스로 분류됨.
