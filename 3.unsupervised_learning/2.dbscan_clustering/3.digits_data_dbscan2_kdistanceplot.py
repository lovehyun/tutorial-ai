import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 1. 데이터 로드
digits = load_digits()
X = digits.data

# 2. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. K-Distance 그래프 (min_samples=5 기준)
min_samples = 5
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# 4. 거리 정렬
distances = np.sort(distances[:, min_samples - 1])

# 5. 그래프 출력
plt.figure(figsize=(8, 6))
plt.plot(distances)
plt.xlabel('데이터 포인트')
plt.ylabel(f'{min_samples}번째 이웃 거리')
plt.title('K-Distance Graph (DBSCAN eps 선택)')
plt.grid()
plt.show()


# ✅ K-Distance Graph 해석
# 그래프가 완만하게 올라가다가
# 오른쪽 끝에서 급격히 치솟는 지점 → 이게 우리가 찾는 elbow (팔꿈치) 포인트입니다.
# ✔️ 이 elbow 지점에서의 y값이 바로 👉 적정 eps 값입니다.

# ✅ 그래프 해석 (지금 그림 기준)
# y축을 보면 약 3 ~ 4 근처에서부터 급격히 상승하고 있어요.

# 현재 그래프의 elbow는 대략 3.5 ~ 4.0 부근으로 해석됩니다.

# ✅ 추천 eps 값
# ✔️ eps = 3.5 ~ 4.0
# ✔️ min_samples = 5 (기존 유지)

# 👉 이 조합으로 다시 DBSCAN을 적용하면 군집 품질이 훨씬 좋아질 가능성이 높습니다.
