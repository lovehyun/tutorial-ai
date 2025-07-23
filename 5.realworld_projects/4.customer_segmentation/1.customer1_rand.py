from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 데이터 생성 (고객 데이터 대용)
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.2, random_state=42)

# 클러스터링
model = KMeans(n_clusters=4)
labels = model.fit_predict(X)

# 시각화를 위한 차원 축소
X_reduced = PCA(n_components=2).fit_transform(X)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='Accent')
plt.title("Customer Segmentation (K-Means + PCA)")
plt.show()
