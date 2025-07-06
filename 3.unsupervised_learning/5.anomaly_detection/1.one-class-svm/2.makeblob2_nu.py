
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import OneClassSVM

# 1. 데이터 생성
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)
outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X_with_outliers = np.vstack([X, outliers])

# 2. 이상치 비율 설정
nu_values = [0.01, 0.05, 0.10]

plt.figure(figsize=(18, 5))

for idx, nu in enumerate(nu_values, 1):
    model = OneClassSVM(kernel='rbf', gamma=0.1, nu=nu)
    model.fit(X_with_outliers)
    y_pred = model.predict(X_with_outliers)

    normal = X_with_outliers[y_pred == 1]
    outliers_pred = X_with_outliers[y_pred == -1]

    plt.subplot(1, 3, idx)
    plt.scatter(normal[:, 0], normal[:, 1], c='blue', label='Normal')
    plt.scatter(outliers_pred[:, 0], outliers_pred[:, 1], c='red', label='Anomaly')
    plt.title(f'One-Class SVM (nu={nu})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

plt.tight_layout()
plt.show()
