# 핵심 파라미터
# 파라미터	설명
# contamination	예상 이상치 비율
# random_state	재현 가능성 확보

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

# 1. 데이터 생성
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)
outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X_with_outliers = np.vstack([X, outliers])

# 2. Isolation Forest 모델
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_with_outliers)
y_pred = model.predict(X_with_outliers)

# 3. 시각화
normal = X_with_outliers[y_pred == 1]
outliers_pred = X_with_outliers[y_pred == -1]

plt.figure(figsize=(8, 6))
plt.scatter(normal[:, 0], normal[:, 1], c='blue', label='Normal')
plt.scatter(outliers_pred[:, 0], outliers_pred[:, 1], c='red', label='Anomaly')
plt.title('Isolation Forest (contamination=0.05)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
