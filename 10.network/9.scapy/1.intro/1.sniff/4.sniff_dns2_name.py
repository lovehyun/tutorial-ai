from scapy.all import sniff, DNSQR

def dns_logger(pkt):
    if pkt.haslayer(DNSQR):
        qname = pkt[DNSQR].qname.decode()
        if "naver.com" in qname:
            print(f"[DNS 요청 감지] {qname}")

sniff(filter="udp port 53", prn=dns_logger, store=False)
