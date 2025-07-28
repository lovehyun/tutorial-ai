# Scapy로 실시간 패킷을 모니터링하면서, 1초마다 초당 수신된 패킷 수(PPS: packets per second)를 출력하는 코드

from scapy.all import sniff
import time

count = 0
start = time.time()

def count_packets(pkt):
    global count, start

    count += 1
    elapsed = time.time() - start
    if elapsed >= 1: # 1초 경과 후
        print(f"{count} packets/sec")
        count = 0
        start = time.time()

sniff(prn=count_packets, store=False)
