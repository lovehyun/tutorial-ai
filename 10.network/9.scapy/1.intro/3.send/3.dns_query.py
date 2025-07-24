# from scapy.all import IP, UDP, DNS, DNSQR
from scapy.all import *

# DNS 요청 생성 및 전송
dns_server = "8.8.8.8"
domain = "www.google.com"
dns_request = IP(dst=dns_server)/UDP(dport=53)/DNS(rd=1, qd=DNSQR(qname=domain))

response = sr1(dns_request, verbose=False, timeout=2)

if response:
    print("응답:", response[DNS].summary())
else:
    print("DNS 응답 없음")
