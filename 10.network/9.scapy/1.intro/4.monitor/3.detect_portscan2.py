from scapy.all import sniff, TCP, IP
from collections import defaultdict
import time

my_ip = "192.168.0.10"  # 본인의 IP
syn_logs = defaultdict(list)  # src_ip → [패킷 정보들]

def detect_syn(pkt):
    if pkt.haslayer(IP) and pkt.haslayer(TCP) and pkt[TCP].flags == 'S':
        src = pkt[IP].src
        dst = pkt[IP].dst
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
        now = time.time()

        if dst != my_ip:
            return  # 나를 대상으로 한 요청만 처리

        log = {
            "timestamp": now,
            "src": src,
            "sport": sport,
            "dst": dst,
            "dport": dport
        }
        syn_logs[src].append(log)

        # 최근 5초 이내 요청만 필터링
        recent_logs = [l for l in syn_logs[src] if now - l["timestamp"] <= 5]
        syn_logs[src] = recent_logs  # 갱신

        print(f"SYN 탐지중: {src}:{sport} → {dst}:{dport}")

        if len(recent_logs) >= 10:
            ports = sorted(set(l["dport"] for l in recent_logs))
            print(f"[🚨 포트 스캔 의심] {src} → {dst}")
            print(f"    최근 5초간 SYN {len(recent_logs)}회")
            print(f"    시도한 포트: {', '.join(map(str, ports))}")
            syn_logs[src].clear()  # 탐지 후 초기화

sniff(filter="tcp", prn=detect_syn, iface="enp0s9", store=False)
