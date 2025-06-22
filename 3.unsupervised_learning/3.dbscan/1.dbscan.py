from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 데이터 생성 (비선형 구조)
X, _ = make_moons(n_samples=100, noise=0.1, random_state=42)

# DBSCAN 적용
model = DBSCAN(eps=0.3, min_samples=5)
y_pred = model.fit_predict(X)

# 시각화
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='plasma')
plt.title("DBSCAN Clustering")
plt.show()
