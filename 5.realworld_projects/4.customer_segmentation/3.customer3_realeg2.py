import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows용 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False      # 마이너스(-) 기호 깨짐 방지

# 1. 의미 있는 예제 고객 데이터 생성
group_1 = pd.DataFrame({'age': [25, 27, 26, 28, 29], 'income': [8000, 8200, 7900, 8100, 8300], 'score': [85, 88, 90, 83, 86]})
group_2 = pd.DataFrame({'age': [22, 24, 23, 21, 25], 'income': [3000, 3100, 2900, 3050, 3200], 'score': [90, 95, 92, 88, 91]})
group_3 = pd.DataFrame({'age': [55, 58, 60, 62, 65], 'income': [9000, 8800, 8700, 8600, 9100], 'score': [20, 25, 18, 22, 21]})
group_4 = pd.DataFrame({'age': [35, 36, 37, 38, 40], 'income': [5000, 5100, 4900, 5200, 5050], 'score': [50, 52, 48, 55, 51]})
df = pd.concat([group_1, group_2, group_3, group_4], ignore_index=True)

# 2. 전처리: 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['age', 'income', 'score']])

# 3. 클러스터링
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df['cluster'] = labels

# 4. PCA로 차원 축소 (2D 시각화를 위한)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)
df['pc1'] = X_reduced[:, 0]
df['pc2'] = X_reduced[:, 1]

# 5. 새로운 고객 정보
new_customer = pd.DataFrame([[30, 7000, 85]], columns=['age', 'income', 'score'])
new_scaled = scaler.transform(new_customer)
new_cluster = kmeans.predict(new_scaled)[0]

# 6. 거리 계산: 새 고객과 각 클러스터 중심점 간 거리
distances = euclidean_distances(new_scaled, kmeans.cluster_centers_)[0]
print(f"\n새 고객은 클러스터 {new_cluster}에 속합니다.")
print("새 고객과 각 클러스터 중심점 간 거리:")
for i, d in enumerate(distances):
    print(f"  - 클러스터 {i}: 거리 {d:.3f}")

# 7. 시각화를 위한 새 고객의 2D 좌표 (PCA 변환)
new_pca = pca.transform(new_scaled)

# 8. 시각화: 기존 고객 + 클러스터 중심 + 새 고객
plt.figure(figsize=(8, 6))

# 기존 고객 점 찍기
for label in sorted(df['cluster'].unique()):
    subset = df[df['cluster'] == label]
    plt.scatter(subset['pc1'], subset['pc2'], label=f"클러스터 {label}", s=100)

# 클러스터 중심 시각화
centroids_2d = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker='X', c='black', s=200, label='중심점')

# 새 고객 점 시각화
plt.scatter(new_pca[:, 0], new_pca[:, 1], marker='*', c='red', s=250, label='새 고객')

# 새 고객과 각 클러스터 중심점 거리 선 그리기
for i, (cx, cy) in enumerate(centroids_2d):
    plt.plot([new_pca[0, 0], cx], [new_pca[0, 1], cy], linestyle='--', color='gray')

plt.title("KMeans 클러스터 + 새 고객 시각화")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend()
plt.grid(True)
plt.show()
