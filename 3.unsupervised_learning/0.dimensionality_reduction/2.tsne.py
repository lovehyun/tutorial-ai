from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 데이터 로드
data = load_iris()
X = data.data
y = data.target

# t-SNE 적용 (2차원으로 축소)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# 시각화
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='Set1')
plt.title("t-SNE (2D projection of Iris)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()
