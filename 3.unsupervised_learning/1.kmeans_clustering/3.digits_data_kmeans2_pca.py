# digits 데이터셋의 기본 정보
# 항목	            값
# 샘플 개수	        1,797개
# 특징 (특성) 수	64개
# 특징 설명	        8 x 8 픽셀 이미지 (흑백)
# 라벨	            0 ~ 9 (10개의 숫자)

# 64차원 → 왜?
# digits 데이터셋은 8 x 8 이미지 = 총 64개의 픽셀 값으로 구성되어 있어요.
# 각 픽셀의 회색 농도(0~16 정수값)를 하나의 특징으로 사용하기 때문에,
# 👉 X.shape = (1797, 64) 입니다.
# 즉, 64차원 데이터입니다.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. 데이터 로드
digits = load_digits()
X = digits.data
y = digits.target  # 실제 라벨 (참고용)

# 2. KMeans 모델 생성 및 학습 (군집 개수: 10)
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X)

# 3. 예측된 군집 라벨
pred_labels = kmeans.labels_

# 4. 차원 축소 (PCA 2차원 시각화)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 5. 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=pred_labels, palette='tab10', legend='full')
plt.title('KMeans Clustering (digits)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='tab10', legend='full')
plt.title('Actual Labels (digits)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Digit')

plt.tight_layout()
plt.show()
