from scapy.all import IP, UDP, Raw, send
import time

target = "8.8.8.8"
port = 53

try:
    while True:
        pkt = IP(dst=target)/UDP(dport=port)/Raw(load="Hello via UDP")
        send(pkt, verbose=False)
        print(f"[UDP] 패킷 전송 → {target}:{port}")
        time.sleep(1)  # 1초 대기
except KeyboardInterrupt:
    print("전송 중단됨.")
