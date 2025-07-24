from scapy.all import IP, ICMP, sr1
import time

target = "8.8.8.8"

while True:
    pkt = IP(dst=target)/ICMP()
    reply = sr1(pkt, timeout=1, verbose=False)
    
    if reply:
        print(f"[ICMP] 응답: {reply.src}")
    else:
        print("[ICMP] 응답 없음")

    time.sleep(1)  # 1초 대기
