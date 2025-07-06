import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. 데이터 로드 및 스케일링
digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. GMM 모델 생성 및 학습
gmm = GaussianMixture(n_components=10, random_state=42)
gmm.fit(X_scaled)

# 3. 군집 레이블 추출
labels = gmm.predict(X_scaled)

# 4. 군집 품질 평가
score = silhouette_score(X_scaled, labels)
print(f"Gaussian Mixture Silhouette Score: {score:.4f}")

# 5. PCA 시각화
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='tab10', legend='full')
plt.title(f'Gaussian Mixture Model (digits)\nSilhouette Score: {score:.4f}')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.show()
