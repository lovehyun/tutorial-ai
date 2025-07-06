from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 데이터 로드
data = load_iris()
X = data.data
y = data.target

# PCA 적용 (2차원으로 축소)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 시각화
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='Set1')
plt.title("PCA (2D projection of Iris)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
