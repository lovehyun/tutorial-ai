from scapy.all import sniff
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np

# LSTM 모델 로드
model = load_model("model_lstm_seq10.h5")

# 패킷 특징 누적용 버퍼
feature_buffer = []

# 실시간 추론 함수
def process_packet(pkt):
    global feature_buffer

    if pkt.haslayer("IP"):
        # 특징 추출: duration은 사용 어려워서 다른 특성으로 대체
        pkt_len = len(pkt)
        src_bytes = len(pkt.payload)
        dst_bytes = pkt_len - src_bytes
        ttl = pkt["IP"].ttl if hasattr(pkt["IP"], 'ttl') else 64

        # 1개 샘플 생성 (duration 대신 TTL)
        sample = [pkt_len, src_bytes, dst_bytes, ttl]
        feature_buffer.append(sample)

        if len(feature_buffer) == 10:
            X = np.array(feature_buffer)

            # 정규화
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_input = np.expand_dims(X_scaled, axis=0)

            # 추론
            y_prob = model.predict(X_input)[0][0]
            y_pred = int(y_prob > 0.5)

            print(f"\n[추론 결과] → {'이상' if y_pred else '정상'} (확률: {y_prob:.3f})")

            # 버퍼 초기화 (슬라이딩 윈도우로 하려면 deque 활용)
            feature_buffer = []

# sniff 시작 (root 권한 필요)
print("실시간 패킷 감지 중... (Ctrl+C로 종료)")
sniff(filter="ip", prn=process_packet, store=0)
