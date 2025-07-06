# 비교 포인트
# contamination	설명
# 0.01	매우 보수적, 이상치를 적게 탐지
# 0.05	적당한 비율, 일반적인 설정
# 0.10	더 많은 데이터를 이상치로 탐지 (오탐지 가능성 증가)

# contamination 값이 커질수록 → 모델이 더 많은 이상치를 잡으려 하고 → 경계가 더 민감해짐.
# contamination 값이 작을수록 → 모델이 보수적으로 판단 → 이상치로 판단하는 데이터가 적어짐.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

# 1. 데이터 생성
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)
outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X_with_outliers = np.vstack([X, outliers])

# 2. 비교할 contamination 값
contamination_values = [0.01, 0.05, 0.10]

plt.figure(figsize=(18, 5))

for idx, contamination in enumerate(contamination_values, 1):
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_with_outliers)
    y_pred = model.predict(X_with_outliers)

    normal = X_with_outliers[y_pred == 1]
    outliers_pred = X_with_outliers[y_pred == -1]

    plt.subplot(1, 3, idx)
    plt.scatter(normal[:, 0], normal[:, 1], c='blue', label='Normal')
    plt.scatter(outliers_pred[:, 0], outliers_pred[:, 1], c='red', label='Anomaly')
    plt.title(f'Isolation Forest (contamination={contamination})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

plt.tight_layout()
plt.show()
