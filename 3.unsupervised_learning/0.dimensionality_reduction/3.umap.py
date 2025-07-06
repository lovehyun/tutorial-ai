# pip install umap-learn

from sklearn.datasets import load_iris
import umap
import matplotlib.pyplot as plt

# 데이터 로드
data = load_iris()
X = data.data
y = data.target

# UMAP 적용
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)

# 시각화
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='Set1')
plt.title("UMAP (2D projection of Iris)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()
