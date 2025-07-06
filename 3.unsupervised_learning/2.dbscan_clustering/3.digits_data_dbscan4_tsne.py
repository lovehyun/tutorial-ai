import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

# 1. 데이터 로드
digits = load_digits()
X = digits.data
y = digits.target

# 2. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. DBSCAN 적용 (최적 파라미터 사용)
dbscan = DBSCAN(eps=6.0, min_samples=3)
dbscan.fit(X_scaled)
labels = dbscan.labels_

# 4. t-SNE로 2차원 축소
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# 5. 시각화
plt.figure(figsize=(12, 6))

# (1) DBSCAN 군집 시각화
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='tab10', legend='full')
plt.title('DBSCAN Clustering (t-SNE)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(title='Cluster', loc='best')

# (2) 실제 라벨 시각화
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='tab10', legend='full')
plt.title('Actual Labels (t-SNE)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(title='Digit', loc='best')

plt.tight_layout()
plt.show()


# KMeans vs DBSCAN 모델 선택 기준
# 비교 항목	            KMeans	                    DBSCAN
# 군집 개수	            미리 지정 가능	            자동 추정
# 군집 모양	원형(구형)  군집에 강함	                임의의 모양도 잘 잡음
# 이상치 처리	        이상치 반영, 모두 군집화	이상치 탐지 가능 (-1)
# 고차원 데이터	        적합	                    거의 부적합
# 밀도 기반	            X (거리 기반)	            O (밀도 기반)
# 고차원 digits	        잘 맞음	                    잘 안 맞음 (지금처럼 실패할 가능성 높음)
# 복잡한 2D	            X	                        잘 맞음

# ✅ digits 데이터는 왜 KMeans가 더 잘 맞을까?
# digits 데이터는 64차원 고차원 벡터입니다.

# DBSCAN은 고차원에서 밀도 차이를 잘 구별하지 못합니다. (차원의 저주 발생)

# ✔️ DBSCAN은
# 👉 2D, 3D처럼 저차원 데이터에서 밀도가 뚜렷한 데이터에 강력합니다.
# 👉 고차원에서는 데이터 간 거리가 모두 비슷해져서 밀도 계산이 잘 안 됩니다.

# 📌 고차원 DBSCAN의 이슈: 차원의 저주
# 고차원에서는 대부분의 점들이 서로 거의 같은 거리로 떨어져 있어서
# 👉 DBSCAN이 밀도 차이를 잘 못 느끼고 군집이 거의 1개로 잡히거나 이상치가 대량 발생합니다.

# ✅ 판단하는 기준 요약
# 판단 질문	답이 Yes → 사용 모델
# 군집 개수를 알고 싶은가?	KMeans
# 군집의 모양이 복잡한가?	DBSCAN
# 이상치를 탐지하고 싶은가?	DBSCAN
# 데이터가 고차원인가?	KMeans
# 데이터가 2D/3D인가?	DBSCAN
# 군집 내 거리 기준이 중요?	KMeans
# 밀도 기반으로 군집화하고 싶은가?	DBSCAN
