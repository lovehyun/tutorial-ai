from scapy.all import sniff, wrpcap

packets = sniff(count=100)
wrpcap("captured.pcap", packets)
print("캡처된 패킷을 captured.pcap 에 저장했습니다.")
