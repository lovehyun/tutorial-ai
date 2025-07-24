from scapy.all import sniff, IP

def packet_callback(pkt):
    if pkt.haslayer(IP):
        print(f"{pkt[IP].src} → {pkt[IP].dst} : {pkt.summary()}")

sniff(filter="ip", prn=packet_callback, store=False)
