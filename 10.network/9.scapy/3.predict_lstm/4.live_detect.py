# 3_live_detect.py
from scapy.all import sniff, IP
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from collections import deque
import time

# 모델 및 스케일러 로드
model = load_model("model_lstm_seq10.h5")
scaler = joblib.load("scaler_lstm.pkl")

# 특징 버퍼 (슬라이딩 윈도우)
feature_buffer = deque(maxlen=10)

def process_packet(pkt):
    if pkt.haslayer("IP"):
        pkt_len = len(pkt)
        src_bytes = len(pkt.payload)
        dst_bytes = pkt_len - src_bytes
        ttl = pkt["IP"].ttl if hasattr(pkt["IP"], 'ttl') else 64

        feature = [pkt_len, src_bytes, dst_bytes, ttl]
        feature_buffer.append(feature)

        if len(feature_buffer) == 10:
            X = np.array(feature_buffer).reshape(1, 10, 4)
            X_scaled = scaler.transform(X[0])
            X_input = X_scaled.reshape(1, 10, 4)

            y_prob = model.predict(X_input, verbose=0)[0][0]
            y_pred = int(y_prob > 0.5)

            print(f"[{time.strftime('%H:%M:%S')}] 추론 결과 → {'🚨 이상' if y_pred else '정상'} (확률: {y_prob:.3f})")

print("▶ 실시간 포트스캔 감지 중... (Ctrl+C로 종료)")
sniff(filter="ip", prn=process_packet, store=0)
