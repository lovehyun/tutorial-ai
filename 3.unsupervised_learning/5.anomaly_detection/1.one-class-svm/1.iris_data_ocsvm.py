# 핵심 파라미터
# nu=0.05 → 이상치 비율 (5% 가정)
# gamma=0.1 → RBF 커널의 반경

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

# 1. 데이터 로드 (정상 데이터만 사용)
iris = load_iris()
X = iris.data

# 2. PCA 2D로 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. One-Class SVM 모델 학습
model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)  # nu: 이상치 비율 (5%)
model.fit(X_pca)

# 4. 예측
y_pred = model.predict(X_pca)

# -1: 이상치, 1: 정상
normal = X_pca[y_pred == 1]
outliers = X_pca[y_pred == -1]

# 5. 시각화
plt.figure(figsize=(8, 6))
plt.scatter(normal[:, 0], normal[:, 1], c='blue', label='Normal')
plt.scatter(outliers[:, 0], outliers[:, 1], c='red', label='Anomaly')
plt.title('Anomaly Detection with One-Class SVM (Iris)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()
