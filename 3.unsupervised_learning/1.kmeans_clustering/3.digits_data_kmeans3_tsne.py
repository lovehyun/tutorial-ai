import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# 1. 데이터 로드
digits = load_digits()
X = digits.data
y = digits.target  # 실제 라벨 (참고용)

# 2. KMeans 군집화 (원본 64차원 데이터)
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X)
pred_labels = kmeans.labels_

# 3. t-SNE 차원 축소 (시각화용)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X)

# 4. 시각화
plt.figure(figsize=(12, 6))

# (1) KMeans 군집 시각화
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=pred_labels, palette='tab10', legend='full')
plt.title('KMeans Clustering (t-SNE 2D)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(title='Cluster')

# (2) 실제 라벨 시각화
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='tab10', legend='full')
plt.title('Actual Labels (t-SNE 2D)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(title='Digit')

plt.tight_layout()
plt.show()
