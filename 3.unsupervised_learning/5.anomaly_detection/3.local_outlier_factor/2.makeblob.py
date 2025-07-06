# Local Outlier Factor (LOF) 개념
# 핵심 원리
# - 국소 밀도(Local Density)를 기준으로 이상치를 탐지하는 알고리즘
# - 내 주변 이웃들의 밀도에 비해 내 밀도가 얼마나 다른지를 비교해서 판단
# - 내 밀도가 주변보다 훨씬 낮으면 → 이상치로 판단
#
# LOF의 특징
# 항목	설명
# 거리 기반	K-최근접 이웃을 활용
# 국소적 판단	주변 이웃들과 상대적으로 비교
# 주로 사용하는 데이터	저차원, 군집 경계가 애매한 데이터
#
# LOF는 주변 이웃이 밀집되어 있는데 나 혼자 멀리 떨어져 있으면 → 이상치로 잡음.

# 핵심 파라미터
# 파라미터	설명
# n_neighbors	이웃의 개수 (보통 10~30)
# contamination	이상치 비율 예상

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor

# 1. 데이터 생성
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)
outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X_with_outliers = np.vstack([X, outliers])

# 2. LOF 모델
model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred = model.fit_predict(X_with_outliers)

# 3. 시각화
normal = X_with_outliers[y_pred == 1]
outliers_pred = X_with_outliers[y_pred == -1]

plt.figure(figsize=(8, 6))
plt.scatter(normal[:, 0], normal[:, 1], c='blue', label='Normal')
plt.scatter(outliers_pred[:, 0], outliers_pred[:, 1], c='red', label='Anomaly')
plt.title('Local Outlier Factor (contamination=0.05)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
