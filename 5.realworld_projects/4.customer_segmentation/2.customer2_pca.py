import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 가상의 고객 데이터 생성
np.random.seed(42)
n_customers = 200
data = {
    'age': np.random.randint(18, 70, n_customers),
    'income': np.random.normal(50000, 15000, n_customers),
    'spending_score': np.random.normal(50, 20, n_customers)
}
df = pd.DataFrame(data)

# 1. 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 2. KMeans 클러스터링
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 3. 차원 축소 (PCA 시각화용)
X_reduced = PCA(n_components=2).fit_transform(X_scaled)

# 4. 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='Accent', s=50)
centers_2d = PCA(n_components=2).fit(X_scaled).transform(kmeans.cluster_centers_)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', marker='X', s=200, label='Centroids')
plt.title("Customer Segmentation with KMeans")
plt.legend()
plt.show()

# 5. 클러스터별 통계 보기
df['cluster'] = labels
print(df.groupby('cluster').mean(numeric_only=True).round(1))
