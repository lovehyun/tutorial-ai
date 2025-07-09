from scapy.all import sniff

def extract_features(pkt):
    if pkt.haslayer("IP"):
        print({
            "src": pkt["IP"].src,
            "dst": pkt["IP"].dst,
            "proto": pkt["IP"].proto,
            "len": len(pkt)
        })

sniff(prn=extract_features, count=10)
