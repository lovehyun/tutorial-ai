from scapy.all import sniff, wrpcap

# 10개의 패킷 캡처
packets = sniff(count=10)

wrpcap("captured.pcap", packets) # writepcap
print("캡처된 패킷을 captured.pcap 에 저장했습니다.")
