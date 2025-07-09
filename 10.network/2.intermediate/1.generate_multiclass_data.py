# 1_generate_multiclass_data.py
import numpy as np
import pandas as pd

np.random.seed(0)
n = 400

# 0: 정상, 1: DoS, 2: Probe
data = []

# 각 데이터 항목
# duration: 연결 시간 (초)
# packet_size: 평균 패킷 크기 (Byte)
# src_bytes: 출발지 바이트 수
# dst_bytes: 목적지 바이트 수

# 정상 트래픽 (label=0)
# duration     ~ N(20, 4)
# packet_size  ~ N(200, 20)
# src_bytes    ~ N(500, 50)
# dst_bytes    ~ N(600, 50)
#
# - duration: 정상 세션은 일정한 시간 유지
# - packet_size: 중간 크기의 일반 패킷
# - src_bytes/dst_bytes: 양쪽 간 데이터 교환이 비교적 균형 있음
# => 현실의 웹/메일 트래픽을 단순화한 모델
for _ in range(n):
    data.append([np.random.normal(20, 4), np.random.normal(200, 20),
                 np.random.normal(500, 50), np.random.normal(600, 50), 0])

# DoS 트래픽 (label=1)
# duration     ~ N(2, 1)
# packet_size  ~ N(1500, 100)
# src_bytes    ~ N(3000, 300)
# dst_bytes    ~ N(100, 30)
# - duration: 짧은 시간에 집중적으로 발생
# - packet_size: 매우 큰 패킷 (MTU 근접)
# - src_bytes: 공격자가 계속 데이터를 보냄 (비정상적으로 큼)
# - dst_bytes: 응답 거의 없음 (서버가 버벅거리거나 차단)
# => 서버를 마비시키는 대량 트래픽 공격의 특성을 반영
for _ in range(n):
    data.append([np.random.normal(2, 1), np.random.normal(1500, 100),
                 np.random.normal(3000, 300), np.random.normal(100, 30), 1])

# Probe 트래픽 (label=2)
# duration     ~ N(10, 3)
# packet_size  ~ N(300, 50)
# src_bytes    ~ N(800, 80)
# dst_bytes    ~ N(500, 60)
# - duration: 중간 정도의 시간 동안 스캔 지속
# - packet_size: 다양하지만 소형/중형 패킷 혼합
# - src_bytes/dst_bytes: 응답을 받아야 하기 때문에 어느 정도 양방향 트래픽 발생
# => 시스템의 열린 포트를 탐색하는 도구(Nmap 등) 특징을 반영
for _ in range(n):
    data.append([np.random.normal(10, 3), np.random.normal(300, 50),
                 np.random.normal(800, 80), np.random.normal(500, 60), 2])

df = pd.DataFrame(data, columns=["duration", "packet_size", "src_bytes", "dst_bytes", "label"])
df.to_csv("network_multiclass.csv", index=False)
print("network_multiclass.csv 저장 완료")
