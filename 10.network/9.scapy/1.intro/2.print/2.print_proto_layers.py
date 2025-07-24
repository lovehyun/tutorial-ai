from scapy.all import sniff, IP, TCP, UDP, ICMP

def extract_features(pkt):
    if pkt.haslayer(IP):
        features = {
            "src": pkt[IP].src,
            "dst": pkt[IP].dst,
            "proto": pkt[IP].proto,
            "len": len(pkt),
        }

        if pkt.haslayer(TCP):
            features.update({
                "sport": pkt[TCP].sport,
                "dport": pkt[TCP].dport,
                "flags": pkt[TCP].flags
            })

        elif pkt.haslayer(UDP):
            features.update({
                "sport": pkt[UDP].sport,
                "dport": pkt[UDP].dport,
            })

        elif pkt.haslayer(ICMP):
            features.update({
                "icmp_type": pkt[ICMP].type,
                "icmp_code": pkt[ICMP].code
            })

        print(features)

sniff(prn=extract_features, count=10, store=False)
