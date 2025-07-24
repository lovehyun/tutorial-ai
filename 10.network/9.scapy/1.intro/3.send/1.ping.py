from scapy.all import IP, ICMP, sr1

# 대상 IP 주소
target = "8.8.8.8"

packet = IP(dst=target) / ICMP()
response = sr1(packet, timeout=2)
# response = sr1(packet, timeout=2, verbose=0)

# sr1() → 응답 하나만 기다림
# sr() → 여러 응답 (요청-응답 페어들)을 기다림
# sniff() → 무작위로 오는 모든 패킷을 다 봄 (매칭 안함)

if response:
    print("응답 받음:", response.summary())
else:
    print("응답 없음")

# | 기호           | 의미                                                    |
# | -------------- | ------------------------------------------------------- |
# | `.` (dot)      | 패킷을 보냈고 **응답을 받았음 (Success)**                 |
# | `*` (asterisk) | 패킷을 보냈지만 **응답을 못 받았음 (Timeout)**            |
# | `X`            | 응답은 받았는데 **잘못된 응답** (예: 포트 unreachable 등 오류 응답)         |
# | `!`            | 특정한 ICMP 오류 메시지를 의미 (예: ICMP Destination Unreachable 등) |
# | `S`            | 패킷 전송 시도 실패 (전송 자체가 안됨)                    |
