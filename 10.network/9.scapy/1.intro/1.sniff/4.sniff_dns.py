from scapy.all import sniff, DNSQR

def dns_monitor(pkt):
    if pkt.haslayer(DNSQR):
        print("DNS 요청:", pkt[DNSQR].qname.decode())

# sniff(filter="udp port 53", prn=dns_monitor, store=False)
# sniff(filter="tcp port 53", prn=dns_monitor, store=False)
sniff(filter="tcp port 53 or udp port 53", prn=dns_monitor, store=False)
