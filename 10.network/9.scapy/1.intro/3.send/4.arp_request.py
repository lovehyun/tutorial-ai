from scapy.all import *

target_ip = "192.168.0.1"

# ARP 요청 전송
arp_request = ARP(pdst=target_ip)
broadcast = Ether(dst="ff:ff:ff:ff:ff:ff")
packet = broadcast/arp_request
answered = srp(packet, timeout=2, verbose=False)[0]

for sent, received in answered:
    print(f"{received.psrc} → {received.hwsrc}")
