# 1_preprocess.py
from scapy.all import rdpcap
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def extract_features(pcap_file):
    packets = rdpcap(pcap_file)
    features = []

    for pkt in packets:
        if pkt.haslayer("IP"):
            pkt_len = len(pkt)
            src_bytes = len(pkt.payload)
            dst_bytes = pkt_len - src_bytes
            ttl = pkt["IP"].ttl if hasattr(pkt["IP"], 'ttl') else 64
            features.append([pkt_len, src_bytes, dst_bytes, ttl])

    return np.array(features)

def create_sequences(data, seq_len=10):
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

# 정상 / 포트스캔 pcap 불러오기
normal = extract_features("normal.pcap")
attack = extract_features("portscan.pcap")

# 정규화
scaler = StandardScaler()
X_all = np.vstack((normal, attack))
scaler.fit(X_all)
joblib.dump(scaler, "scaler_lstm.pkl")

# 시퀀스 생성
X_normal = create_sequences(scaler.transform(normal), 10)
X_attack = create_sequences(scaler.transform(attack), 10)

y_normal = np.zeros(len(X_normal))
y_attack = np.ones(len(X_attack))

X = np.concatenate((X_normal, X_attack), axis=0)
y = np.concatenate((y_normal, y_attack), axis=0)

# 저장
np.savez("dataset_lstm.npz", X=X, y=y)
