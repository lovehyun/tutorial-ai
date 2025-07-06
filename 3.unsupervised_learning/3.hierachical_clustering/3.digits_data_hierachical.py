# 항목	설명
# linkage='ward'	최소 분산 기준으로 군집화
# truncate_mode='level'	digits 데이터는 샘플이 많아서 덴드로그램 일부만 시각화
# n_clusters=10	digits는 정답 라벨이 10개이므로 군집 개수를 10개로 설정
# PCA	시각화용 2차원 축소

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

# 1. 데이터 로드
digits = load_digits()
X = digits.data
y = digits.target

# 2. 덴드로그램 그리기 (거리 계산)
linked = linkage(X, method='ward')

plt.figure(figsize=(15, 6))

dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=False,
        #    color_threshold=200  # 적당히 커트라인을 직접 지정 (matplotlib이 거리 상위 70% 정도의 큰 분리 구간만 색상을 다르게 칠해줍니다.)
)

# dendrogram(linked,
#            truncate_mode='level',  # 너무 복잡하니까 일부만 보기
#            p=30,  # 최대 30개 레벨만 표시
#            orientation='top',
#            distance_sort='descending',
#            show_leaf_counts=False)

# dendrogram(linked,
#            truncate_mode='lastp',
#            p=30,
#            color_threshold=150,  # 이 값을 조절하면 색상 구간이 바뀜
#            orientation='top',
#            distance_sort='descending',
#            show_leaf_counts=True)

plt.title('Hierarchical Clustering Dendrogram (digits)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# 3. 계층적 군집화 (군집 개수 10개로 설정)
cluster = AgglomerativeClustering(n_clusters=10, linkage='ward')
labels = cluster.fit_predict(X)

# 4. PCA로 2차원 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 5. 군집 품질 평가
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.4f}")

# 6. 시각화
plt.figure(figsize=(12, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='tab10', legend='full')
plt.title('Agglomerative Clustering (digits)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.show()

# 7. 군집 분포 확인
unique_clusters = np.unique(labels)
print(f"발견된 군집 개수: {len(unique_clusters)}")


# 기대 결과
# - 덴드로그램: 군집 형성 단계가 계단식으로 보임 (복잡하지만 잘 나타남)
# - 시각화: 군집이 어느 정도 분리된 모습이 보임
# - digits 데이터는 복잡하기 때문에 완벽한 군집 분리는 어려울 수 있지만 PCA로 2D 축소 후 유사한 패턴이 보이면 성공입니다.
