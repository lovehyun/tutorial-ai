# pip install scapy-http
from scapy.all import sniff, ARP, IP, TCP, UDP, ICMP, DNS 
from scapy.layers.http import HTTPRequest, HTTPResponse
from threading import Thread
from collections import Counter
import time

class PacketCollector:
    def __init__(self):
        self.packets = []
        self.protocol_counter = Counter()
        self.ip_counter = Counter()
        self.packet_id = 0 # 순번 처리 카운터

    def start(self):
        t = Thread(target=self._sniff_loop, daemon=True)
        t.start()

    def _sniff_loop(self):
        sniff(prn=self._handle_packet, store=False)

    def _handle_packet(self, pkt):
        self.packet_id += 1
        ts = time.strftime('%H:%M:%S')
        
        sport = dport = '-'
        if pkt.haslayer(ARP):
            proto = 'ARP'
        elif pkt.haslayer(ICMP):
            proto = 'ICMP'
        elif pkt.haslayer(TCP):
            proto = 'TCP'
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
            if pkt.haslayer(HTTPRequest) or pkt.haslayer(HTTPResponse):
                proto = 'HTTP'
        elif pkt.haslayer(UDP):
            proto = 'UDP'
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
            if pkt.haslayer(DNS):
                proto = 'DNS'
        else:
            proto = 'OTHER'

        sip = pkt[IP].src if pkt.haslayer(IP) else '-'
        dip = pkt[IP].dst if pkt.haslayer(IP) else '-'
        summary = pkt.summary()

        self.protocol_counter[proto] += 1
        self.ip_counter[sip] += 1
        self.packets.append({
            'id': self.packet_id,
            'time': ts,
            'proto': proto,
            'sip': sip,
            'sport': sport,
            'dip': dip,
            'dport': dport,
            'summary': summary
        })
        
        if len(self.packets) > 100:
            self.packets.pop(0)

    def get_stats(self):
        return {
            'logs': list(reversed(self.packets)),  # 최신순
            'protocols': dict(self.protocol_counter),
            'sources': dict(self.ip_counter)
        }
