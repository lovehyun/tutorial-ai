from scapy.all import sniff, TCP, IP
from collections import defaultdict

# (출발지 IP, 목적지 포트) 조합별 SYN 패킷 수를 저장할 딕셔너리
syn_counts = defaultdict(int)

def detect_syn(pkt):
    # TCP 패킷이고, flags가 'S' (SYN)만 있는 경우
    if pkt.haslayer(TCP) and pkt[TCP].flags == 'S':
        key = (pkt[IP].src, pkt[TCP].dport)
        
        # 해당 키 조합의 SYN 요청 수 증가
        syn_counts[key] += 1
        
        # 특정 IP가 동일 포트로 SYN을 10번 넘게 보냈다면 경고 출력
        if syn_counts[key] > 10:
            print(f"[!] 포트 스캔 의심: {pkt[IP].src} → {pkt[TCP].dport}")

sniff(filter="tcp", prn=detect_syn, store=False)
