from scapy.all import IP, ICMP, sr1

packet = IP(dst="8.8.8.8") / ICMP()
response = sr1(packet, timeout=1)

if response:
    print("응답 받음:", response.summary())
else:
    print("응답 없음")
