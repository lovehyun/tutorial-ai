from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 데이터 생성
X, _ = make_blobs(n_samples=50, centers=3, random_state=42)

# 계층적 클러스터링
Z = linkage(X, method='ward')

# 덴드로그램 그리기
plt.figure(figsize=(8, 4))
dendrogram(Z)
plt.title("Hierarchical Clustering (Dendrogram)")
plt.show()
