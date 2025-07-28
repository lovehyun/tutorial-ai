from scapy.all import sniff, DNSQR

# 1. 도메인 하나 감지
# def dns_logger(pkt):
#     if pkt.haslayer(DNSQR):
#         qname = pkt[DNSQR].qname.decode()
#         if "naver.com" in qname:
#             print(f"[DNS 요청 감지] {qname}")
#
# sniff(filter="tcp port 53 or udp port 53", prn=dns_logger, store=False)


# 2. 블랙리스트 도메인 목록 감지
blacklist = ["naver.com", "google.com", "youtube.com"]
def dns_logger(pkt):
    if pkt.haslayer(DNSQR):
        qname = pkt[DNSQR].qname.decode()
        for domain in blacklist:
            if domain in qname:
                print(f"[🚫 차단 도메인 감지] {qname}")
                break  # 중복 출력 방지

# UDP 53 포트에서 DNS 요청 감시
sniff(filter="tcp port 53 or udp port 53", prn=dns_logger, store=False)
