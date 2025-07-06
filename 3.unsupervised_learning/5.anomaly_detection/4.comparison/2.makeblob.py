from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# 정상 데이터 생성
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)

# 이상치 생성
outliers = np.random.uniform(low=-6, high=6, size=(20, 2))

# 데이터 결합
X_with_outliers = np.vstack([X, outliers])

# 시각화
plt.scatter(X_with_outliers[:, 0], X_with_outliers[:, 1])
plt.title('Synthetic Dataset with Outliers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# 3가지 모델 정의
models = {
    'One-Class SVM': OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05),
    'Isolation Forest': IsolationForest(contamination=0.05, random_state=42),
    'Local Outlier Factor': LocalOutlierFactor(n_neighbors=20, contamination=0.05)
}

plt.figure(figsize=(18, 5))

for idx, (name, model) in enumerate(models.items(), 1):
    if name == 'Local Outlier Factor':
        y_pred = model.fit_predict(X_with_outliers)
    else:
        model.fit(X_with_outliers)
        y_pred = model.predict(X_with_outliers)

    # -1: 이상치, 1: 정상
    normal = X_with_outliers[y_pred == 1]
    outliers = X_with_outliers[y_pred == -1]

    plt.subplot(1, 3, idx)
    plt.scatter(normal[:, 0], normal[:, 1], c='blue', label='Normal')
    plt.scatter(outliers[:, 0], outliers[:, 1], c='red', label='Anomaly')
    plt.title(name)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

plt.tight_layout()
plt.show()
