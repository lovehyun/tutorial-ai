from scapy.all import sniff, TCP, IP
from collections import defaultdict

syn_counts = defaultdict(int)

def detect_syn(pkt):
    if pkt.haslayer(TCP) and pkt[TCP].flags == 'S':  # SYN 패킷
        key = (pkt[IP].src, pkt[TCP].dport)
        syn_counts[key] += 1
        if syn_counts[key] > 10:
            print(f"[!] 포트 스캔 의심: {pkt[IP].src} → {pkt[TCP].dport}")

sniff(filter="tcp", prn=detect_syn, store=False)
