# 1_generate_data.py
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 500

# | 특성          | 정상 트래픽 (label=0)    | 이상 트래픽 (label=1)     | 의미 분석                                    |
# | ------------- | ----------------------- | ------------------------ | ------------------------------------------- |
# | `duration`    | 평균 20초, 표준편차 5초  | 평균 2초, 표준편차 1초     | → 악성 트래픽은 짧은 시간 안에 집중적으로 발생 |
# | `packet_size` | 평균 200B, 표준편차 30B  | 평균 1500B, 표준편차 200B | → 악성은 비정상적으로 큰 패킷 (예: DoS, DDoS) |
# | `src_bytes`   | 평균 500B, 표준편차 100B | 평균 3000B, 표준편차 500B | → 공격자가 많은 데이터를 보내는 패턴           |
# | `dst_bytes`   | 평균 600B, 표준편차 120B | 평균 100B, 표준편차 50B   | → 수신 측은 적은 응답 (예: 서버가 처리 못함)   |
# | `label`       | 0                       | 1                        | → 이진 분류용 레이블                         |

normal_data = {
    'duration': np.random.normal(20, 5, n_samples),
    'packet_size': np.random.normal(200, 30, n_samples),
    'src_bytes': np.random.normal(500, 100, n_samples),
    'dst_bytes': np.random.normal(600, 120, n_samples),
    'label': 0
}

attack_data = {
    'duration': np.random.normal(2, 1, n_samples),
    'packet_size': np.random.normal(1500, 200, n_samples),
    'src_bytes': np.random.normal(3000, 500, n_samples),
    'dst_bytes': np.random.normal(100, 50, n_samples),
    'label': 1
}

df = pd.concat([pd.DataFrame(normal_data), pd.DataFrame(attack_data)], ignore_index=True)
df.to_csv("network_data.csv", index=False)
print("network_data.csv 생성 완료 (1000개 샘플)")
