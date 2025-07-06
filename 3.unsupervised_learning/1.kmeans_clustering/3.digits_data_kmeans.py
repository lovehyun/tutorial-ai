from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드
digits = load_digits()
X = digits.data
y = digits.target

# 2. 스케일링 (KMeans는 스케일링 없이도 가능하지만, 보통 스케일링을 추천)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. KMeans 군집화
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_scaled)

# 4. 결과 출력
print("클러스터 라벨:", set(kmeans.labels_))
print("클러스터 분포:", {label: list(kmeans.labels_).count(label) for label in set(kmeans.labels_)})

# 5. 주의: 시각화는 불가능 (64차원 → 2D 변환 안 했기 때문)


# KMeans 군집 번호는 실제 라벨과 순서가 다르기 때문에 "최빈값 매칭" 필요
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.stats import mode

# 6군집 라벨 추출
cluster_labels = kmeans.labels_

# 7. 군집 번호 vs 실제 라벨 매칭
matched_labels = np.zeros_like(cluster_labels)

for cluster in range(10):
    mask = (cluster_labels == cluster)
    matched_labels[mask] = mode(y[mask])[0]

# 8. 정확도 계산
accuracy = accuracy_score(y, matched_labels)
print(f"KMeans 정확도 (PCA 없이): {accuracy:.4f}")
