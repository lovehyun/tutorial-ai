# pip install scapy

from scapy.all import sniff

def show_packet(pkt):
    print(pkt.summary())

sniff(filter="tcp", prn=show_packet, count=5)
