from scapy.all import sniff

def extract_features(pkt):
    if pkt.haslayer("IP"):
        print({
            "src": pkt["IP"].src,        # 출발지 IP
            "dst": pkt["IP"].dst,        # 목적지 IP
            "proto": pkt["IP"].proto,    # 프로토콜 번호 (예: TCP=6, UDP=17, ICMP=1)
            "len": len(pkt)              # 전체 패킷 길이 (bytes)
        })

sniff(prn=extract_features, count=10)
