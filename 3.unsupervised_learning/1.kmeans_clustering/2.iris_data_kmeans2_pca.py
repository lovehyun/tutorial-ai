# 핵심 포인트
# 항목	        설명
# 데이터	    iris (라벨은 비지도 학습에서는 참고용)
# 군집 개수	    n_clusters=3
# 차원 축소	    PCA로 2차원 변환 후 시각화
# 출력	        군집 라벨 분포 + 시각화

# 추가 실습
# - n_clusters 값을 2, 4로 변경해서 비교해 보기
# - 다른 데이터셋 (예: digits)로 실습 확장
# - Elbow Method로 최적의 k 찾기 실습

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target  # 실제 라벨 (KMeans는 이걸 사용하지 않음)

# 2. KMeans 모델 생성 및 학습 (군집 개수: 3)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 3. 예측된 군집 라벨
pred_labels = kmeans.labels_

# 4. 실제 라벨과 예측 라벨 비교
df = pd.DataFrame(X, columns=iris.feature_names)
df['Cluster'] = pred_labels
df['Actual'] = y

print("\n클러스터 할당 개수:")
print(df['Cluster'].value_counts())

# 5. 차원 축소 (PCA 2차원 시각화)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# 6. KMeans 군집 시각화
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='deep')
plt.title('KMeans Clustering (PCA 2D)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.grid()
plt.show()

# 7. 실제 라벨 시각화
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Actual', data=df, palette='Set2')
plt.title('Actual Labels (PCA 2D)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Actual')
plt.grid()
plt.show()
