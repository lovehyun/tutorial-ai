from scapy.all import sniff, wrpcap, rdpcap

# 캡처 후 저장
# packets = sniff(count=10)
# wrpcap("captured.pcap", packets)

# 다시 읽기
pkts = rdpcap("captured.pcap")
for p in pkts:
    print(p.summary())
