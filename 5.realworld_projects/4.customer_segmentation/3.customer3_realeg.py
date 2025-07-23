import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows용 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False      # 마이너스(-) 기호 깨짐 방지

# 1. 의미 있는 가상의 고객 데이터 구성
# 고객 속성: 나이(age), 수입(income), 소비성향(spending score)
data = {
    # 고소득 고소비 고객 (프리미엄 고객)
    'age':     [25, 27, 26, 28, 29],
    'income':  [8000, 8200, 7900, 8100, 8300],
    'score':   [85, 88, 90, 83, 86]
}
df1 = pd.DataFrame(data)

data = {
    # 저소득 고소비 고객 (젊은 소비층)
    'age':     [22, 24, 23, 21, 25],
    'income':  [3000, 3100, 2900, 3050, 3200],
    'score':   [90, 95, 92, 88, 91]
}
df2 = pd.DataFrame(data)

data = {
    # 고소득 저소비 고객 (은퇴/저소비층)
    'age':     [55, 58, 60, 62, 65],
    'income':  [9000, 8800, 8700, 8600, 9100],
    'score':   [20, 25, 18, 22, 21]
}
df3 = pd.DataFrame(data)

data = {
    # 중간 수입/중간 소비 고객 (평균 소비자)
    'age':     [35, 36, 37, 38, 40],
    'income':  [5000, 5100, 4900, 5200, 5050],
    'score':   [50, 52, 48, 55, 51]
}
df4 = pd.DataFrame(data)

# 모든 고객 데이터를 하나로 합치기
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 확인용 출력
print("원본 고객 데이터 예시:\n", df.head())

# 2. 표준화: 각 feature의 단위가 다르기 때문에 정규화 필요
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['age', 'income', 'score']])  # DataFrame 컬럼 이름 있음.

# 3. KMeans 클러스터링 수행
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df['cluster'] = labels  # 결과를 데이터프레임에 추가

# 4. PCA로 차원 축소 (시각화를 위해 3D → 2D)
X_reduced = PCA(n_components=2).fit_transform(X_scaled)
df['pc1'] = X_reduced[:, 0]
df['pc2'] = X_reduced[:, 1]

# 5. 시각화 (색으로 클러스터 구분)
plt.figure(figsize=(8, 6))
for label in sorted(df['cluster'].unique()):
    subset = df[df['cluster'] == label]
    plt.scatter(subset['pc1'], subset['pc2'], label=f"클러스터 {label}", s=100)

# 클러스터 중심도 시각화
centroids_2d = PCA(n_components=2).fit(X_scaled).transform(kmeans.cluster_centers_)
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker='X', c='black', s=200, label='중심')

plt.title("KMeans 고객 클러스터링 (PCA 시각화)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()

# 6. 클러스터별 고객 특성 평균 보기
# → 비슷한 고객군이 어떤 성향인지 분석 가능
cluster_summary = df.groupby('cluster')[['age', 'income', 'score']].mean().round(1)
print("\n클러스터별 고객군 평균 특성:")
print(cluster_summary)

# 7. 해석 예시 (결과를 본 후 사후 해석)
for cluster_id, row in cluster_summary.iterrows():
    print(f"\n[클러스터 {cluster_id}]")
    if row['income'] > 7000 and row['score'] > 80:
        print("→ 고소득 & 고소비: 프리미엄 고객층")
    elif row['income'] < 4000 and row['score'] > 80:
        print("→ 저소득 & 고소비: 젊은 소비층")
    elif row['income'] > 8000 and row['score'] < 30:
        print("→ 고소득 & 저소비: 은퇴/절약 고객층")
    else:
        print("→ 평균적인 고객층")

# 8. 새로운 고객: 나이 30, 수입 7000, 소비 점수 85
# new_customer = np.array([[30, 7000, 85]])  # numpy array (컬럼 이름 없음)
new_customer = pd.DataFrame([[30, 7000, 85]], columns=['age', 'income', 'score'])

# 기존과 같은 전처리 (정규화)
new_scaled = scaler.transform(new_customer)  # 경고 발생 ( UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names )

# KMeans로 클러스터 예측
cluster_id = kmeans.predict(new_scaled)[0]
print(f"\n\n새 고객은 클러스터 {cluster_id}에 속합니다.")

# 해당 군집의 특성 출력
summary = df[df['cluster'] == cluster_id][['age', 'income', 'score']].mean().round(1)
print("\n이 고객이 속한 그룹의 평균 특성:")
print(summary)
