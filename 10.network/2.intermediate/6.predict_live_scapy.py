# 5_predict_live.py
from scapy.all import sniff, IP, TCP, Raw
import joblib
import time
import pandas as pd

# 모델 불러오기 (Pipeline 포함: StandardScaler + RandomForest)

# 이 전 모델에서 make_pipeline() 사용했기 때문에...
# 이 경우 StandardScaler → RandomForest 흐름이 하나의 객체에 포함되고,
# joblib.dump()는 그 전체 파이프라인 객체를 그대로 저장합니다.

model = joblib.load("models/rf_model.pkl")

# 시간 기록용 변수
last_time = {}
duration_dict = {} # 동일한 송신 IP → 수신 IP 조합이면 이전 시간과의 차이(duration) 를 구해서 특징으로 넣고 있음

print("실시간 패킷 모니터링 시작... (Ctrl+C로 종료)")

def extract_features(packet):
    if not packet.haslayer(IP):
        return None

    ip = packet[IP]
    key = (ip.src, ip.dst)

    # duration 측정 (이전 시간과 현재 시간의 차이)
    now = time.time()
    prev = last_time.get(key, now)
    duration = now - prev
    last_time[key] = now

    # Raw payload
    raw_len = len(packet[Raw].load) if packet.haslayer(Raw) else 0

    # 특징 추출
    features = {
        'duration': duration,
        'packet_size': len(packet),
        'src_bytes': raw_len if ip.src < ip.dst else 0,
        'dst_bytes': raw_len if ip.src > ip.dst else 0
    }
    return features

def predict_and_display(features):
    input_df = pd.DataFrame([features])  # ← dict → DataFrame으로 변환
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][pred]

    label_map = {0: "Normal", 1: "DoS", 2: "Probe"}
    print(f"예측: {label_map[pred]} | 확률: {prob:.3f} | {features}")

# 실시간 패킷 처리
def process_packet(packet):
    features = extract_features(packet)
    if features:
        predict_and_display(features)

sniff(prn=process_packet, store=False)
