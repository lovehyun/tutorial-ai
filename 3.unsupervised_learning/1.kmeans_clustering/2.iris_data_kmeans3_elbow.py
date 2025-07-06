# Elbow Method 해석:
# - 그래프가 급격히 꺾이는 지점이 최적의 k입니다.
# - iris 데이터셋에서는 보통 k = 3 근처가 가장 적합합니다.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# 1. 데이터 로드
iris = load_iris()
X = iris.data

# 2. 군집 개수별 inertia 계산
inertia_list = []
k_list = range(1, 11)  # k=1~10까지 테스트

for k in k_list:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia_list.append(kmeans.inertia_)  # inertia: 군집 내 거리의 제곱합

# 3. Elbow 그래프 시각화
plt.figure(figsize=(8, 6))
plt.plot(k_list, inertia_list, 'o-', color='blue')
plt.xlabel('군집 개수 (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method (iris)')
plt.grid()
plt.xticks(k_list)
plt.show()
