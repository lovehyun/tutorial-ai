from scapy.all import get_if_list
from scapy.all import sniff, IP
from threading import Thread
import time

class PacketCollector:
    def __init__(self):
        self.packets = []

    def start(self):
        print("[*] 패킷 수집 시작")
        t = Thread(target=self._sniff_loop, daemon=True)
        t.start()

    def _sniff_loop(self):
        sniff(prn=self._handle_packet, store=False)

    def _handle_packet(self, pkt):
        if pkt.haslayer(IP):
            self.packets.append({
                'sip': pkt[IP].src,
                'dip': pkt[IP].dst,
                'proto': pkt[IP].proto
            })

            # print(f"[+] {time.strftime('%H:%M:%S')} | {pkt.summary()}")

            if len(self.packets) > 100:
                self.packets.pop(0)

    def get_packets(self):
        return list(self.packets)

collector = PacketCollector()
