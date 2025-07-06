# 항목	설명
# 데이터	digits 데이터 (64차원)
# 스케일링	DBSCAN과 KMeans에서 필수
# 비교 모델	KMeans, DBSCAN, AgglomerativeClustering
# 평가 지표	Silhouette Score
# 시각화	PCA 2차원 축소 후 군집 시각화

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. 데이터 로드 및 스케일링
digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 군집화 모델 정의
models = {
    'KMeans': KMeans(n_clusters=10, random_state=42),
    'DBSCAN': DBSCAN(eps=6.0, min_samples=4),
    'Agglomerative': AgglomerativeClustering(n_clusters=10, linkage='ward')
}

# 3. 결과 저장
results = {}

for name, model in models.items():
    labels = model.fit_predict(X_scaled)

    # Silhouette Score는 군집이 2개 이상일 때만 계산
    if len(set(labels)) > 1:
        score = silhouette_score(X_scaled, labels)
    else:
        score = -1  # 군집이 1개로만 묶인 경우

    results[name] = {
        'labels': labels,
        'score': score
    }

    print(f"{name} - Silhouette Score: {score:.4f}")

# 4. PCA 시각화
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(18, 5))

for idx, (name, result) in enumerate(results.items(), 1):
    plt.subplot(1, 3, idx)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=result['labels'], palette='tab10', legend='full')
    plt.title(f"{name}\nSilhouette Score: {result['score']:.4f}")
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend(title='Cluster', loc='best')

plt.tight_layout()
plt.show()


# ✅ 각 모델 성능 요약
# 군집화 모델	Silhouette Score	해석
# KMeans	0.1356	군집 구분은 나쁘지 않지만 일부 겹침
# DBSCAN	0.2704	Silhouette Score는 가장 높지만 군집 개수 매우 적음
# Agglomerative	0.1253	KMeans와 유사한 군집 구분, 밀집도 약간 더 낮음

# ✅ 결과 해석 (핵심)
# ✔️ DBSCAN
# Silhouette Score는 가장 높지만, 군집이 4~5개로만 분리됨.
# 실제 digits는 10개 군집이 목표인데, DBSCAN은 거의 대부분을 Cluster 0으로 묶어버림.
# 👉 밀도 기반 군집화가 잘 안 맞는 데이터라는 신호.
# ✔️ KMeans
# 군집 개수: 정확히 10개로 잘 나눔
# Silhouette Score는 낮지만 digits 데이터 자체가 복잡해서 어느 정도 이해 가능.
# 👉 실제 분류 목적에는 KMeans가 더 적합.
# ✔️ Agglomerative
# KMeans와 군집 구조가 거의 비슷함.
# Silhouette Score는 KMeans보다 조금 더 낮음.
# 👉 KMeans보다 약간 더 덜 응집된 군집.

# ✅ 결론
# 평가	결과
# 실루엣 점수	DBSCAN이 가장 높음
# 실제 군집 개수 맞춤	KMeans가 가장 적합
# 데이터 분포 반영	DBSCAN은 대부분을 한 군집으로 몰아서 부적합

# ✔️ 비지도 군집화에서 가장 좋은 선택:
# 👉 KMeans가 digits 데이터에는 가장 현실적인 선택입니다.
# ✔️ DBSCAN은 군집 개수를 맞추기 힘들고, digits처럼 복잡한 고차원 데이터에서는 밀도 차이를 잘 구별하지 못하는 한계가 그대로 드러났어요.

# ✅ 요약
# 모델	적합성
# KMeans	✅ 가장 적합
# DBSCAN	❌ 군집 개수 부족
# Agglomerative	⭕ 비교적 적합 (KMeans 유사)
