import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 2. 덴드로그램 그리기 (거리 계산)
linked = linkage(X, method='ward')  # ward: 유클리드 거리 기반 최소 분산

plt.figure(figsize=(12, 6))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (iris)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# 3. 계층적 군집화 모델 (군집 개수 3으로 고정)
cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward') # ward면 metric 생략가능 ('ward' → 최소 분산 기준)
labels = cluster.fit_predict(X)

# 4. 시각화 (PCA 차원 축소)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(12, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set1', legend='full')
plt.title('Agglomerative Clustering (iris)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.show()


# 덴드로그램에서 군집 개수는 내가 직접 커트라인을 조정해서 바꿀 수 있음.
# iris 데이터는 실제로 3개의 군집으로 잘 나뉨.
# 시각화 결과가 실제 라벨과 꽤 비슷하게 나오는지 확인하면 됩니다.
