# digits 데이터는 64차원이라 eps를 0.5로 하면 대부분 이상치로 나옵니다.
# 👉 eps를 3 ~ 5 정도로 크게 시작하는 게 좋습니다.
#
# DBSCAN은 라벨을 전혀 모르는 상태에서 군집화하기 때문에
# 정답 라벨과 비교하는 정확도 계산은 비지도 평가에 적합하지 않습니다.
#
# 👉 대신 silhouette_score를 사용하면 군집화 품질을 객관적으로 볼 수 있습니다.

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
dbscan = DBSCAN(eps=4, min_samples=5)  # eps는 digits에서는 조금 크게 시작하는 것이 좋음
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


# DBSCAN 군집 개수: 26
# DBSCAN 이상치 데이터 개수: 770
# Silhouette Score: -0.0587

# 현재 결과는 군집이 과도하게 많이 생성됐고,
# Silhouette Score가 음수라는 건 👉 군집이 잘못 형성됐다는 신호입니다.
# 이건 eps와 min_samples를 잘못 설정했기 때문에 발생하는 자연스러운 현상이에요.
