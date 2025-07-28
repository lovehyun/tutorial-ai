# sudo setcap cap_net_raw,cap_net_admin+eip /usr/bin/python3.12

from scapy.all import IP, TCP, sr1

target = "scanme.nmap.org"
ports = [22, 80, 443]
# ports = range(20, 31) # 20~30까지

for port in ports:
    pkt = IP(dst=target) / TCP(dport=port, flags='S')
    resp = sr1(pkt, timeout=1, verbose=0)

    if resp and resp.haslayer(TCP) and resp[TCP].flags == 0x12:
        print(f"포트 {port} 열림")
    else:
        print(f"포트 {port} 닫힘 또는 필터링")
