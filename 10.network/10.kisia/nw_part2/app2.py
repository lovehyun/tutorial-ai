from flask import Flask, render_template, jsonify
import threading
import time
import scapy.all as scapy
import pandas as pd
import joblib
from collections import deque, Counter
import numpy as np

app = Flask(__name__)

# 모델 및 스케일러 로딩
model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")
abnormal_cluster = joblib.load("models/abnormal_cluster.pkl")

# 실시간 저장소
latest_results = deque(maxlen=200)        # 최근 탐지 결과
packet_count_history = deque(maxlen=500)  # 최근 timestamp 기록

# JSON 직렬화 보조 함수
def convert_to_builtin(obj):
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    return obj

# 패킷에서 Feature 추출 (임시값 - 실제 트래픽 분석 로직 필요)
def extract_features_from_packet(pkt):
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
            proto = ip_layer.proto

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
            'sip': sip,
            'sport': sport,
            'dip': dip,
            'dport': dport,
            'protocol': proto,
            'size': size,
            'info': "; ".join(info)
        }
    except Exception as e:
        return {
            'sip': '-', 'sport': '-', 'dip': '-', 'dport': '-',
            'protocol': 'ERR', 'size': 0, 'info': str(e)
        }

# 실시간 패킷 수집 및 분석
def packet_sniffer():
    def process(pkt):
        timestamp = int(time.time())
        packet_count_history.append(timestamp)

        features_raw = extract_features_from_packet(pkt)
        metadata = extract_packet_metadata(pkt)

        X = pd.DataFrame([features_raw], columns=features)
        X_scaled = scaler.transform(X)

        cluster = model.predict(X_scaled)[0]
        is_abnormal = (cluster == abnormal_cluster)

        latest_results.append({
            'timestamp': timestamp,
            'cluster': int(cluster),
            'abnormal': is_abnormal,
            'features': features_raw,
            'meta': metadata
        })

    scapy.sniff(prn=process, store=0, filter="tcp port 80")

# 상태 조회 API
@app.route('/status')
def status():
    now = int(time.time())
    one_sec_ago = now - 1
    one_min_ago = now - 60

    # 최근 1분 이내의 결과만 필터링
    recent_results = [d for d in latest_results if d['timestamp'] >= one_min_ago]

    # PPS / MPS
    pps = sum(1 for t in packet_count_history if t == one_sec_ago)
    mps = sum(1 for t in packet_count_history if t >= one_min_ago)

    # 시간대별 패킷 수
    rate_counter = Counter(t for t in packet_count_history if t >= one_min_ago)
    recent_timestamps = sorted(rate_counter.keys())[-20:]
    packet_rate = [{'timestamp': ts, 'count': rate_counter[ts]} for ts in recent_timestamps]

    # 클러스터별 통계
    cluster_stats = {}
    for d in recent_results:
        cluster = d['cluster']
        is_abnormal = d['abnormal']
        if cluster not in cluster_stats:
            cluster_stats[cluster] = {'count': 0, 'abnormal': 0}
        cluster_stats[cluster]['count'] += 1
        if is_abnormal:
            cluster_stats[cluster]['abnormal'] += 1

    # 직렬화
    serializable = [
        {k: convert_to_builtin(v) if k != 'features' else v for k, v in d.items()}
        for d in recent_results
    ]

    return jsonify({
        'packets_per_second': pps,
        'packets_per_minute': mps,
        'results': serializable,
        'packet_rate': packet_rate,
        'cluster_stats': cluster_stats
    })

@app.route('/')
def home():
    return render_template('index2.html')

# 백그라운드 스레드 시작
threading.Thread(target=packet_sniffer, daemon=True).start()

if __name__ == '__main__':
    app.run(debug=True)
