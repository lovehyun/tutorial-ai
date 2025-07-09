import numpy as np
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

# 2. 학습에서 사용한 StandardScaler로 정규화
# 실제 환경에선 scaler를 joblib으로 저장했다가 불러오는 게 맞음
scaler = StandardScaler()
scaler.fit(new_data)  # 여기서는 new_data 자체에 fit (정확하진 않음)
scaled = scaler.transform(new_data)

# 3. LSTM 입력 형태로 변환
X_input = np.expand_dims(scaled, axis=0)  # shape = (1, 10, 4)

# 4. 모델 불러오기
model = load_model("model_lstm_seq10.h5")

# 5. 추론
y_prob = model.predict(X_input)[0][0]
y_pred = int(y_prob > 0.5)

print(f"예측 결과: {'이상' if y_pred else '정상'} (확률: {y_prob:.3f})")
