# gamma	결과
# 작으면	경계가 넓어짐 → 이상치를 잘 안 잡음
# 크면	경계가 좁아짐 → 과하게 이상치로 판단할 수도 있음

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import OneClassSVM

# 데이터 생성
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)
outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X_with_outliers = np.vstack([X, outliers])

# 비교할 gamma 값
gamma_values = [0.01, 0.1, 1.0]

plt.figure(figsize=(18, 5))

for idx, gamma in enumerate(gamma_values, 1):
    model = OneClassSVM(kernel='rbf', gamma=gamma, nu=0.05)
    model.fit(X_with_outliers)
    y_pred = model.predict(X_with_outliers)

    normal = X_with_outliers[y_pred == 1]
    outliers_pred = X_with_outliers[y_pred == -1]

    plt.subplot(1, 3, idx)
    plt.scatter(normal[:, 0], normal[:, 1], c='blue', label='Normal')
    plt.scatter(outliers_pred[:, 0], outliers_pred[:, 1], c='red', label='Anomaly')
    plt.title(f'One-Class SVM (gamma={gamma})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

plt.tight_layout()
plt.show()
