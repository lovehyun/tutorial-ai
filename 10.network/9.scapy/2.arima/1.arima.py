# pip install statsmodels
from scapy.all import sniff
import time
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import platform

if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # Windows: 맑은 고딕
elif platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Linux (예: Colab, Ubuntu 등)
    plt.rc('font', family='NanumGothic')  # 또는 설치된 폰트명

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 1. 초단위 패킷 수 수집
packet_counts = []
interval = 1  # 초 단위
duration = 60  # 총 수집 시간 (초)

def count_packets(pkt_list):
    return len(pkt_list)

print("패킷 수집 중...")

for i in range(duration):
    start_time = time.time()
    pkts = sniff(timeout=interval)
    count = count_packets(pkts)
    packet_counts.append(count)
    print(f"{i+1}초: {count} packets")

# 2. 시계열 데이터프레임 생성
ts = pd.Series(packet_counts)
ts.index = pd.date_range(start='2025-01-01', periods=len(ts), freq='s')  # 초 단위

# 3. ARIMA 모델 학습 (간단한 설정)
model = ARIMA(ts, order=(2, 1, 2))  # (p, d, q)는 실제에 맞게 튜닝 필요
model_fit = model.fit()

# 4. 미래 10초 예측 (초단위 예측)
forecast = model_fit.get_forecast(steps=10)
predicted_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# 5. 시각화
plt.figure(figsize=(12, 6))
plt.plot(ts, label='Observed')
plt.plot(predicted_mean.index, predicted_mean, color='red', label='Forecast')
plt.fill_between(predicted_mean.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='pink', alpha=0.3, label='Confidence Interval')
plt.title('ARIMA 기반 실시간 패킷 수 예측 (초단위)')
plt.xlabel('Time')
plt.ylabel('Packets per Second')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
