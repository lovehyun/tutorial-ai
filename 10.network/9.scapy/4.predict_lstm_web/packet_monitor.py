from scapy.all import sniff
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from socketio import Client
import numpy as np

model = load_model("model_lstm_seq10.h5")
sio = Client()
sio.connect("http://localhost:5000")

buffer = []

def process_packet(pkt):
    global buffer

    if pkt.haslayer("IP"):
        pkt_len = len(pkt)
        src_bytes = len(pkt.payload)
        dst_bytes = pkt_len - src_bytes
        ttl = pkt["IP"].ttl if hasattr(pkt["IP"], 'ttl') else 64
        sample = [pkt_len, src_bytes, dst_bytes, ttl]
        buffer.append(sample)

        if len(buffer) == 10:
            X = np.array(buffer)
            X_scaled = StandardScaler().fit_transform(X)
            X_input = np.expand_dims(X_scaled, axis=0)
            y_prob = model.predict(X_input)[0][0]
            y_pred = int(y_prob > 0.5)

            sio.emit("result_from_packet", {
                "label": "이상" if y_pred else "정상",
                "score": round(float(y_prob), 3)
            })
            buffer = []

print("패킷 감지 중... (Ctrl+C 종료)")
sniff(filter="ip", prn=process_packet, store=0)
