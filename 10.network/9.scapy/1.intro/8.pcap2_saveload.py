from scapy.all import sniff, wrpcap, rdpcap

# 캡처 후 저장
packets = sniff(count=10)
wrpcap("test.pcap", packets)

# 다시 읽기
pkts = rdpcap("test.pcap")
for p in pkts:
    print(p.summary())
