from scapy.all import *

for ip in range(1, 11):
    target = f"192.168.1.{ip}"
    pkt = IP(dst=target)/ICMP()
    res = sr1(pkt, timeout=1, verbose=0)
    if res:
        print(f"[응답] {target}")
    else:
        print(f"[없음] {target}")

