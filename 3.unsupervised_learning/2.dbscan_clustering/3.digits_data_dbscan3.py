import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# 1. 데이터 로드
digits = load_digits()
X = digits.data
y = digits.target

# 2. 스케일링 (DBSCAN은 거리 기반 → 필수)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. DBSCAN 모델 생성 및 학습
dbscan = DBSCAN(eps=6.0, min_samples=3)  # eps는 digits에서는 조금 크게 시작하는 것이 좋음
dbscan.fit(X_scaled)

# 4. 결과
labels = dbscan.labels_  # -1은 이상치

# 5. 군집 개수와 이상치 개수
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"DBSCAN 군집 개수: {n_clusters}")
print(f"DBSCAN 이상치 데이터 개수: {n_noise}")

# 6. 군집 품질 평가 (라벨 필요 없음)
if n_clusters > 1:
    silhouette = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {silhouette:.4f}")
else:
    print("군집이 1개 또는 모두 이상치입니다. Silhouette Score를 계산할 수 없습니다.")

# eps=3.0, samples=5
# DBSCAN 군집 개수: 13
# DBSCAN 이상치 데이터 개수: 1618
# Silhouette Score: -0.2332

# eps=5.0, samples=5
# DBSCAN 군집 개수: 4
# DBSCAN 이상치 데이터 개수: 207
# Silhouette Score: -0.0094

# eps=6.0, samples=3
# DBSCAN 군집 개수: 4
# DBSCAN 이상치 데이터 개수: 56
# Silhouette Score: 0.2528

# eps를 6.0으로 키우고, min_samples를 3으로 완화한 것이 아주 좋은 선택이었어요.
# ✔️ 이상치가 과도하지 않고, 군집이 너무 많지도 않은 좋은 밸런스입니다.
# ✔️ Silhouette Score가 0.25면 비지도 군집화에서는 꽤 괜찮은 품질이에요.
# (비지도 학습에서는 0.2~0.3만 되어도 실무에서는 자주 씁니다.)
