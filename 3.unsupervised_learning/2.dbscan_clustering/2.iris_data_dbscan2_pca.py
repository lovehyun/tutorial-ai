import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 2. 스케일링 (DBSCAN은 거리 기반 → 스케일링 필수)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. DBSCAN 모델 생성 및 학습
dbscan = DBSCAN(eps=0.6, min_samples=5)
dbscan.fit(X_scaled)

# 4. 예측된 군집 라벨
labels = dbscan.labels_  # -1은 이상치 (noise)

# 5. 차원 축소 (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 6. 시각화 (DBSCAN 결과)
plt.figure(figsize=(12, 6))

# (1) DBSCAN 결과
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set1', legend='full')
plt.title('DBSCAN Clustering')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')

# (2) 실제 라벨
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='Set2', legend='full')
plt.title('Actual Labels')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Actual')

plt.tight_layout()
plt.show()

# 7. 군집 분포 확인
print("DBSCAN 클러스터 라벨:", np.unique(labels))
