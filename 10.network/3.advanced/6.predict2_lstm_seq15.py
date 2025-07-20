import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# 1. 새로운 15초 시퀀스 데이터 생성 (정상처럼 보이는 트래픽 예시)
np.random.seed(42)
sequence_length = 15
new_data = np.column_stack([
    np.random.normal(20, 4, sequence_length),      # duration
    np.random.normal(200, 20, sequence_length),    # packet_size
    np.random.normal(500, 50, sequence_length),    # src_bytes
    np.random.normal(600, 50, sequence_length),    # dst_bytes
])  # shape = (15, 4)

# 2. 정규화 (학습 시 저장한 스케일러 사용)
scaler = joblib.load("scaler_seq15.pkl")
scaled_data = scaler.transform(new_data)

# 3. 모델 입력 형태로 변환 (1, 15, 4)
X_input = np.expand_dims(scaled_data, axis=0)

# 4. 모델 불러오기
model = load_model("model_lstm_seq15.h5")

# 5. 예측
y_prob = model.predict(X_input)[0][0]
y_pred = int(y_prob > 0.5)

# 6. 출력
print(f"예측 결과: {'이상' if y_pred else '정상'} (확률: {y_prob:.3f})")
