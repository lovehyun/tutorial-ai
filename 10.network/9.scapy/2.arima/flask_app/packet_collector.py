from scapy.all import sniff
import pandas as pd
from threading import Thread
from statsmodels.tsa.arima.model import ARIMA
import time
from datetime import datetime
import warnings

# 불필요한 경고 제거
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)

class PacketMonitor:
    def __init__(self):
        self.df = pd.DataFrame(columns=['timestamp', 'count'])

    def start(self):
        t = Thread(target=self._sniff_loop, daemon=True)
        t.start()

    def _sniff_loop(self):
        while True:
            count = len(sniff(timeout=1))
            now = pd.to_datetime(datetime.now())
            self.df.loc[len(self.df)] = [now, count]
            time.sleep(0.1)  # 너무 과도한 루프 방지

    def get_resampled(self, rule):
        # '1s', '1min', '1h' 등의 규칙을 받아서 리샘플링
        df = self.df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])  # 강제 변환
        df = df.set_index('timestamp')
        return df.resample(rule).sum().dropna()

    def get_forecast(self, data, steps=10):
        if len(data) < 30:  # 최소 길이 보장
            return list(data[-10:]), [], [], []
        try:
            model = ARIMA(data, order=(2, 1, 2), enforce_stationarity=False)
            fit = model.fit()
            forecast = fit.get_forecast(steps=steps)
            return (
                list(data[-10:]),
                list(forecast.predicted_mean),
                list(forecast.conf_int().iloc[:, 0]),
                list(forecast.conf_int().iloc[:, 1])
            )
        except Exception as e:
            print("ARIMA 예측 실패:", e)
            return list(data[-10:]), [], [], []
