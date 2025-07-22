import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1. 시퀀스 설정
sequence_length = 10
features = ["duration", "packet_size", "src_bytes", "dst_bytes"]

# 2. 저장된 스케일러 및 모델 불러오기
scaler = joblib.load("scaler_seq10.pkl")
model = load_model("model_lstm_seq10.h5")

# 3. 정상 트래픽 시퀀스 생성 함수
def generate_normal_sequence():
    return np.column_stack([
        np.random.normal(20, 4, sequence_length),     # duration
        np.random.normal(200, 20, sequence_length),   # packet_size
        np.random.normal(500, 50, sequence_length),   # src_bytes
        np.random.normal(600, 50, sequence_length),   # dst_bytes
    ])

# 4. 비정상 트래픽 시퀀스 생성 함수
def generate_anomalous_sequence():
    return np.column_stack([
        np.random.normal(150, 50, sequence_length),    # duration: 매우 큼
        np.random.normal(1500, 300, sequence_length),  # packet_size: 매우 큼
        np.random.normal(10000, 500, sequence_length), # src_bytes: 비정상적으로 큼
        np.random.normal(50, 10, sequence_length),     # dst_bytes: 비정상적으로 작음
    ])

# 5. 예측 함수
def predict_sequence(seq, label):
    scaled = scaler.transform(seq)
    X_input = np.expand_dims(scaled, axis=0)
    y_prob = model.predict(X_input)[0][0]
    y_pred = int(y_prob > 0.5)
    print(f"[{label}] 예측 결과: {'이상' if y_pred else '정상'} (확률: {y_prob:.3f})")

# 6. 실행
np.random.seed(42)  # 반복 가능하도록 시드 고정

# 정상 시퀀스 예측
normal_seq = generate_normal_sequence()
predict_sequence(normal_seq, "정상 시퀀스")

# 비정상 시퀀스 예측
anomalous_seq = generate_anomalous_sequence()
predict_sequence(anomalous_seq, "비정상 시퀀스")
