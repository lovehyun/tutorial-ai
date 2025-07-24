from scapy.all import sniff
import time

count = 0
start_time = time.time()

# 레벨 기준 (필요시 조정 가능)
def get_level(packets_per_sec):
    if packets_per_sec <= 10:
        return "🔹 Level 1: Normal"
    elif packets_per_sec <= 50:
        return "⚠️ Level 2: Warning"
    else:
        return "🚨 Level 3: Critical"

# 패킷 처리 함수
def count_packets(pkt):
    global count, start_time
    count += 1
    now = time.time()

    if now - start_time >= 1.0:  # 매 1초마다
        pps = count
        level = get_level(pps)
        print(f"[{time.strftime('%H:%M:%S')}] {pps} packets/sec → {level}")
        count = 0
        start_time = now

# 실시간 모니터링 시작
print("실시간 패킷 감시 시작... (Ctrl+C로 중단)")
sniff(prn=count_packets, store=False)
