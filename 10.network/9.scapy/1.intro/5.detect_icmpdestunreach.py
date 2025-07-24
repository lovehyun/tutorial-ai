from scapy.all import sniff, ICMP

def icmp_watch(pkt):
    if pkt.haslayer(ICMP) and pkt[ICMP].type == 3:
        print(f"[!] ICMP 목적지 도달 불가: {pkt.summary()}")

sniff(filter="icmp", prn=icmp_watch, store=False)
