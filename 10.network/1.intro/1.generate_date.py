# 1_generate_data.py
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 500

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
