# pip install scapy

from scapy.all import sniff

def show_packet(pkt):
    print(pkt.summary())
    # pkt.show()

sniff(filter="tcp", prn=show_packet, count=5)
# count 없이 실행하면 Ctrl+C로 중단할 때까지 계속 캡처
# sniff(filter="tcp", prn=show_packet)
