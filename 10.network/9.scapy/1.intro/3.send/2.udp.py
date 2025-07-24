# from scapy.all import IP, UDP, Raw, send
from scapy.all import *

target = "8.8.8.8"
port = 53

# UDP 패킷 생성 및 전송
udp = IP(dst=target)/UDP(dport=port)/Raw(load="Hello via UDP")
send(udp)
print("UDP 패킷 전송 완료")
