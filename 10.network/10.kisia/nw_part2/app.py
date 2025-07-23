from flask import Flask, render_template, jsonify
import threading
import time
import scapy.all as scapy
import pandas as pd
import joblib
from collections import deque
import numpy as np

app = Flask(__name__)

# 모델 및 스케일러 로드
model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")

cols = ['method_cnt','method_post','protocol_1_0','status_major','status_404','status_499',
        'status_cnt','path_same','path_xmlrpc','ua_cnt','has_payload','req_cnt_per_hour']

# 최근 결과 저장소
latest_results = deque(maxlen=500)

def extract_features_from_packet(pkt):
    # 단순 예제 (향후 실트래픽 분석 시 수정 가능)
    return {
        'method_cnt': 1,
        'method_post': 1.0,
        'protocol_1_0': 0,
        'status_major': 0.0,
        'status_404': 0.0,
        'status_499': 0,
        'status_cnt': 1,
        'path_same': 1.0,
        'path_xmlrpc': 0.0,
        'ua_cnt': 1,
        'has_payload': 1,
        'req_cnt_per_hour': 50
    }

def extract_packet_metadata(pkt):
    try:
        proto = "OTHER"
        sip, sport, dip, dport = "-", "-", "-", "-"
        info = []

        if pkt.haslayer(scapy.IP):
            ip_layer = pkt[scapy.IP]
            sip = ip_layer.src
            dip = ip_layer.dst

        if pkt.haslayer(scapy.TCP):
            tcp_layer = pkt[scapy.TCP]
            sport = tcp_layer.sport
            dport = tcp_layer.dport
            proto = "TCP"
            info.append(f"Flags={tcp_layer.flags}")

        elif pkt.haslayer(scapy.UDP):
            udp_layer = pkt[scapy.UDP]
            sport = udp_layer.sport
            dport = udp_layer.dport
            proto = "UDP"

        elif pkt.haslayer(scapy.ICMP):
            proto = "ICMP"

        size = len(pkt)
        return {
            'timestamp': int(time.time()),
            'protocol': proto,
            'sip': sip,
            'sport': sport,
            'dip': dip,
            'dport': dport,
            'size': size,
            'info': "; ".join(info)
        }
    except:
        return {
            'timestamp': int(time.time()),
            'protocol': "ERR", 'sip': '-', 'sport': '-', 'dip': '-', 'dport': '-', 'size': 0, 'info': 'ParseError'
        }

# 실시간 패킷 감지
def packet_sniffer():
    def process(pkt):
        meta = extract_packet_metadata(pkt)
        features = extract_features_from_packet(pkt)
        X = pd.DataFrame([features], columns=cols)
        X_scaled = scaler.transform(X)
        cluster = model.predict(X_scaled)[0]
        is_abnormal = (cluster == 2)

        latest_results.append({
            **meta,
            'cluster': int(cluster),
            'abnormal': is_abnormal
        })

    scapy.sniff(prn=process, store=0)

@app.route('/')
def home():
    return render_template('index.html')

def convert_to_builtin(obj):
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    return obj

@app.route('/logs')
def logs():
    now = int(time.time())
    recent = [r for r in latest_results if r['timestamp'] >= now - 5]

    # numpy 타입 → 기본 타입으로 변환
    serializable = []
    for r in recent:
        fixed = {k: convert_to_builtin(v) for k, v in r.items()}
        serializable.append(fixed)

    return jsonify(serializable)


# 백그라운드로 sniff 시작
threading.Thread(target=packet_sniffer, daemon=True).start()

if __name__ == '__main__':
    app.run(debug=True)
