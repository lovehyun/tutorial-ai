import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 2. 스케일링 (필수)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. DBSCAN 군집화 (PCA 없이)
dbscan = DBSCAN(eps=0.6, min_samples=5)
dbscan.fit(X_scaled)

# 4. 예측된 군집 라벨
labels = dbscan.labels_

# 5. 군집 분포 출력
print("DBSCAN 클러스터 라벨:", np.unique(labels))
print("클러스터 개수:", len(set(labels)) - (1 if -1 in labels else 0))
print("이상치 데이터 개수:", list(labels).count(-1))
