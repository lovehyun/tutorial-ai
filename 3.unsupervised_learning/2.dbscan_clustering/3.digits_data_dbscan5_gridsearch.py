# 설명
# eps: 3.0 ~ 7.0 까지 0.5 단위로 반복
# min_samples: 3 ~ 5로 반복
# 군집이 1개로만 형성되면 평가 제외
# 가장 높은 Silhouette Score 조합을 최종 선택
#
# 참고
# DBSCAN은 k-means처럼 "정답 개수"가 없기 때문에
# 👉 정확도(accuracy) 평가 대신 Silhouette Score를 사용하는 것이 가장 적합합니다.
# 완전 자동화된 eps 튜닝 라이브러리는 없고
# 👉 현재처럼 직접 반복문을 돌려서 점수를 비교하는 것이 일반적인 방법이에요.

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# 1. 데이터 로드
digits = load_digits()
X = digits.data

# 2. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 파라미터 탐색 범위 설정
eps_values = np.arange(3.0, 7.0, 0.5)
min_samples_values = range(3, 6)

best_score = -1
best_params = {}

# 4. Grid Search 반복
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

        # 군집이 1개 또는 모두 이상치면 평가 생략
        if len(set(labels)) <= 1 or len(set(labels)) - (1 if -1 in labels else 0) < 2:
            continue

        score = silhouette_score(X_scaled, labels)
        print(f"eps: {eps}, min_samples: {min_samples}, Silhouette Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_params = {'eps': eps, 'min_samples': min_samples}

# 5. 최적 파라미터 출력
if best_params:
    print(f"\n최적 파라미터: eps = {best_params['eps']}, min_samples = {best_params['min_samples']}")
    print(f"최고 Silhouette Score: {best_score:.4f}")
else:
    print("적절한 군집이 형성되지 않았습니다.")


# eps: 3.0, min_samples: 3, Silhouette Score: -0.2869
# eps: 3.0, min_samples: 4, Silhouette Score: -0.2312
# eps: 3.0, min_samples: 5, Silhouette Score: -0.2332
# eps: 3.5, min_samples: 3, Silhouette Score: -0.1614
# eps: 3.5, min_samples: 4, Silhouette Score: -0.1627
# eps: 3.5, min_samples: 5, Silhouette Score: -0.1866
# eps: 4.0, min_samples: 3, Silhouette Score: -0.0970
# eps: 4.0, min_samples: 4, Silhouette Score: -0.0650
# eps: 4.0, min_samples: 5, Silhouette Score: -0.0587
# eps: 4.5, min_samples: 3, Silhouette Score: -0.0632
# eps: 4.5, min_samples: 4, Silhouette Score: -0.0601
# eps: 4.5, min_samples: 5, Silhouette Score: -0.0602
# eps: 5.0, min_samples: 3, Silhouette Score: -0.0748
# eps: 5.0, min_samples: 4, Silhouette Score: 0.0108
# eps: 5.0, min_samples: 5, Silhouette Score: -0.0094
# eps: 5.5, min_samples: 3, Silhouette Score: 0.2435
# eps: 5.5, min_samples: 4, Silhouette Score: 0.0230
# eps: 5.5, min_samples: 5, Silhouette Score: 0.0039
# eps: 6.0, min_samples: 3, Silhouette Score: 0.2528
# eps: 6.0, min_samples: 4, Silhouette Score: 0.2704
# eps: 6.5, min_samples: 3, Silhouette Score: 0.2643
# eps: 6.5, min_samples: 4, Silhouette Score: 0.2603
#
# 최적 파라미터: eps = 6.0, min_samples = 4
# 최고 Silhouette Score: 0.2704

