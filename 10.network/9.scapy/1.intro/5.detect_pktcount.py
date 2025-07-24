from scapy.all import sniff
import time

count = 0
start = time.time()

def count_packets(pkt):
    global count
    count += 1
    elapsed = time.time() - start
    if elapsed >= 1:
        print(f"{count} packets/sec")
        count = 0
        globals()['start'] = time.time()

sniff(prn=count_packets, store=False)
