# 5_predict_live.py
from scapy.all import sniff
import joblib
import pandas as pd
import numpy as np

# 모델 로드
model = joblib.load("rf_model.pkl")
features = ['duration', 'packet_size', 'src_bytes', 'dst_bytes']  # CSV 기준

def extract_features(pkt):
    if pkt.haslayer("IP"):
        pkt_len = len(pkt)
        src_bytes = len(pkt.payload)
        dst_bytes = pkt_len - src_bytes
        ttl = pkt["IP"].ttl if hasattr(pkt["IP"], 'ttl') else 64

        # duration은 아직 구현 안됨 → 임의 고정값
        duration = 1

        return [duration, pkt_len, src_bytes, dst_bytes]
    return None

def process_packet(pkt):
    sample = extract_features(pkt)
    if sample:
        X = pd.DataFrame([sample], columns=features)  # features는 ['duration', 'packet_size', 'src_bytes', 'dst_bytes']

        # 예측 및 확률
        y_pred = model.predict(X)[0]
        y_prob = model.predict_proba(X)[0][1]  # class=1 (이상)의 확률
        
        print(f"[예측 결과] → {'이상 트래픽' if y_pred == 1 else '정상 트래픽'} | {sample} | 확률: {y_prob:.3f}")


print("실시간 패킷 감지 중... (Ctrl+C로 종료)")
sniff(filter="ip", prn=process_packet, store=0)


# [예측 결과] → (정상 트래픽 | 이상 트래픽) | [duration, packet_size, src_bytes, dst_bytes]

# | 항목          | 의미                                                         |
# | ------------- | ----------------------------------------------------------- |
# | `예측 결과`    | 머신러닝 모델이 해당 샘플을 **정상(0)** 또는 **이상(1)** 으로 판단한 결과 |
# | `duration`    | 패킷 흐름 지속 시간 (단위: 초) – 여기선 단순히 1로 고정         |
# | `packet_size` | 전체 패킷 길이 (bytes)                                       |
# | `src_bytes`   | 송신 측이 보낸 데이터량                                       |
# | `dst_bytes`   | 수신 측이 응답한 데이터량                                     |
# | `ttl`         | TTL (Time To Live) – 패킷 생존시간 (여기선 `14`로 고정된 값)   |
