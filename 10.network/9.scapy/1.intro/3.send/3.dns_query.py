from scapy.all import IP, UDP, DNS, DNSQR, sr1
# from scapy.all import *

# DNS 요청 생성 및 전송
dns_server = "8.8.8.8"
domain = "www.google.com"

# rd는 Recursion Desired의 약자입니다.
# DNS 요청을 보낼 때, "필요하면 다른 DNS 서버까지 재귀적으로 조회해줘"라는 요청을 나타냅니다
#  - 1이면 재귀적 질의를 요청하는 것이고,
#  - 0이면 비재귀적 질의만 허용하겠다는 의미입니다.
dns_request = IP(dst=dns_server) / UDP(dport=53) / DNS(rd=1, qd=DNSQR(qname=domain))

response = sr1(dns_request, verbose=False, timeout=2)

if response:
    print("응답:", response[DNS].summary())
    
    for i in range(response[DNS].ancount):
        answer = response[DNS].an[i]
        if answer.type == 1:  # Type 1 = A record (IPv4)
            print("도메인 IP 주소:", answer.rdata)
else:
    print("DNS 응답 없음")
