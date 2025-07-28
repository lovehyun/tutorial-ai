from scapy.all import sniff, get_if_list
from scapy.all import get_working_ifaces

print(get_if_list())

for iface in get_working_ifaces():
    print(f"{iface.name} - {iface.description}")

def show_packet(pkt):
    print(pkt.summary())
    # pkt.show()

sniff(filter="tcp", iface="eth0", prn=show_packet, count=5)
