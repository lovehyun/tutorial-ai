from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 더미 데이터 생성
X, _ = make_blobs(n_samples=100, centers=3, cluster_std=1.0, random_state=42)

# KMeans 클러스터링
model = KMeans(n_clusters=3, random_state=42)
y_pred = model.fit_predict(X)

# 시각화
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title("K-Means Clustering")
plt.show()
