from scapy.all import sniff, Raw

def get_http(pkt):
    if pkt.haslayer(Raw):
        payload = pkt[Raw].load.decode(errors="ignore")
        if "HTTP" in payload:
            print("HTTP 요청 감지!")
            print(payload)

sniff(filter="tcp port 80", prn=get_http, store=False)
