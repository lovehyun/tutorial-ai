# from scapy.all import IP, UDP, Raw, send
from scapy.all import *

target = "8.8.8.8"
port = 53

# UDP 패킷 생성 및 전송
udp = IP(dst=target)/UDP(dport=port)/Raw(load="Hello via UDP")
send(udp)
print("UDP 패킷 전송 완료")


# nc -u -l 8000
# 기본적으로 한 번 수신하고 종료됨 (UDP라 계속 열고 있음)

# while true; do nc -u -l 8000; done
